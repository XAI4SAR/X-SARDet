from uncertainty.inference.Bayesian_inference import Approximate_Bayesian_Detector, apply_dropout
import argparse
from mmdet.apis import init_detector
import cv2
import math
import numpy as np
from multiprocessing import Pool
import tqdm
import os
from shapely.geometry import Polygon,MultiPoint
import matplotlib.pyplot as plt
import pycocotools.mask as maskUtils
import mmdet.datasets.SAR.data_holders as data_holders
from mmdet.datasets.SAR.pdq import _calc_fg_loss,_calc_bg_loss,_calc_spatial_qual

_HEATMAP_THRESH = 0.0027

def read_result(result_path):
    detections = open(result_path,'r').readlines()
    result = []
    for det in detections:
        det = eval(det)
        result.append(det)
    return result

def obb2hbb(box):
    if len(box) == 4:
        return box
    elif len(box) == 8:
        xmin = min(box[2*i] for i in range(4))
        xmax = max(box[2*i] for i in range(4))
        ymin = min(box[2*i+1] for i in range(4))
        ymax = max(box[2*i+1] for i in range(4))
        return [xmin,ymin,xmax,ymax]

class GroundTruthInstance(object):
    def __init__(self, segmentation_mask, true_class_label, bounding_box, num_pixels=None):
        self.segmentation_mask = segmentation_mask
        self.class_label = true_class_label
        self.bounding_box = bounding_box
        # Calculate the number of pixels based on segmentation mask if unprovided at initialisation
        if num_pixels is not None and num_pixels > 0:
            self.num_pixels = num_pixels
        else:
            self.num_pixels = np.count_nonzero(segmentation_mask)

def bbox2mask(box,img_info):
    if len(box)==8:
        segm = box
    else:
        segm = [box[0],box[1],box[2],box[1],box[2],box[3],box[0],box[3]]
    h, w = img_info
    rles = maskUtils.frPyObjects([segm], h, w)
    rle = maskUtils.merge(rles)
    m = maskUtils.decode(rle)
    return m

class DetectionInstance(object):
    def __init__(self, class_list, heatmap=None):
        self._heatmap = heatmap
        self.class_list = class_list

    def calc_heatmap(self, img_size):
        return self._heatmap

    def get_max_class(self):
        return np.argmax(self.class_list)

    def get_max_score(self):
        return np.amax(self.class_list)

class PBoxDetInst(DetectionInstance):
    def __init__(self, class_list, box, covs):
        super(PBoxDetInst, self).__init__(class_list)
        self.box = box
        self.covs = covs

    def calc_heatmap(self, img_size):
        # get all covs in format (y,x) to match matrix ordering
        covs2 = [np.flipud(np.fliplr(cov)) for cov in self.covs]
        prob1 = data_holders.gen_single_heatmap(img_size, [self.box[1], self.box[0]], covs2[0])
        prob2 = data_holders.gen_single_heatmap(img_size, [max(img_size[0] - (self.box[3] + 1),0), max(img_size[1] - (self.box[2] + 1),0)],
                                   np.array(covs2[1]).T)
        # flip left-right and up-down to provide probability in from bottom-right corner
        prob2 = np.fliplr(np.flipud(prob2))
        # generate final heatmap
        heatmap = prob1 * prob2
        # Hack to enforce that there are no pixels with probs greater than 1 due to floating point errors
        heatmap[heatmap > 1] = 1
        heatmap[heatmap < _HEATMAP_THRESH] = 0

        return heatmap
    
def response_of_masked(new_detections,target_detections,mask,index,save_dir_for_mask):
    target_uncertaintys = [target_detection['cls-uncertainty'] for target_detection in target_detections]
    target_instances = [GroundTruthInstance(bbox2mask(obb2hbb(det['bbox'].copy()),AS.image_size),
                                                [0],
                                                obb2hbb(det['bbox'].copy())
                                                ) for det in target_detections]
    
    similarity_between_targets_and_all_detections = []
    uncertainty_of_all_detection = []
     
    for det in new_detections:
        score = det['score']
        cls_uncertainty = det['cls-uncertainty']
        det_instance = PBoxDetInst(
                class_list=[det['score']],
                box=obb2hbb(det['bbox']),
                covs=det['reg-covs']
            )
        
        similarity_in_LS = [AS.spatial_quality(target_instance,det_instance).item() for target_instance in target_instances]
        similarity_between_targets_and_one_detection = similarity_in_LS
        
        similarity_between_targets_and_all_detections.append(similarity_between_targets_and_one_detection)
        uncertainty_of_all_detection.append(cls_uncertainty)
    try:
        index_of_approdet = np.argmax(similarity_between_targets_and_all_detections,axis=0)
        uncertainty_of_appro = [uncertainty_of_all_detection[i] for i in index_of_approdet]
        sptial_quility_of_appro = [similarity_between_targets_and_all_detections[i][idx] for idx,i in enumerate(index_of_approdet)]

        scores_for_one_mask = [sptial_quility_of_appro[i]*\
            AS.similarity_in_cls_uncertainty(target_uncertaintys[i],uncertainty_of_appro[i]) for i in range(len(target_uncertaintys))]
        
        scores_for_one_mask = np.array(scores_for_one_mask)
        np.save(os.path.join(save_dir_for_mask,'mask-{0}.npy'.format(str(index))),mask)
        np.save(os.path.join(save_dir_for_mask,'weight-{0}.npy'.format(str(index))),scores_for_one_mask)  
    except:
        scores_for_one_mask = [0] * len(target_instances)
        np.save(os.path.join(save_dir_for_mask,'mask-{0}.npy'.format(str(index))),mask)
        np.save(os.path.join(save_dir_for_mask,'weight-{0}.npy'.format(str(index))),scores_for_one_mask)
            
                       
class attribution_analysis():
    def __init__(self, image, targets, mask_type, save_dir, n_masks=2000):
        self.image = image
        self.image_size = image.shape[:2]

        ## target ##
        self.targets = targets

        ## sampling paremeter ##
        assert mask_type in ['local']

        self.mask_function = self.generate_mask_for_one_target

        self.n_masks = n_masks
        self.prob_thresh = 0.5
        self.grid_size = (7, 7)

        self.save_dir = save_dir
        self.sampling_nums = 10


    def similarity_in_cls_uncertainty(self,target_uncertainty,uncertainty):
        return  4 * np.log10 (uncertainty/target_uncertainty)


    def calculate_iou(self,A,B):
        if len(A) == len(B) == 4: #HBB
            area1 = max((A[2]-A[0])*(A[3]-A[1]),1e-7)
            area2 = max((B[2]-B[0])*(B[3]-B[1]),1e-7)
            xx1 = max(A[0],B[0])
            yy1 = max(A[1],B[1])
            xx2 = min(A[2],B[2])
            yy2 = min(A[3],B[3])
            iter_area = max(xx2-xx1,0) * max(yy2-yy1,0)
            IoU = iter_area / (area1+area2-iter_area)  
        else:
            A = [(A[2*i],A[2*i+1]) for i in range(int(len(A)/2))]
            B = [(B[2*i],B[2*i+1]) for i in range(int(len(B)/2))]
            ploy_A = Polygon(A).convex_hull
            ploy_B = Polygon(B).convex_hull   
            union_ploy = np.concatenate((A,B))         
            if not ploy_A.intersects(ploy_B):
                IoU = 0.0
            else: 
                union_area = MultiPoint(union_ploy).convex_hull.area
                iter_area = ploy_A.intersection(ploy_B).area
                if union_area == 0:
                    IoU = 0.0
                else:
                    IoU = float(iter_area)/float(union_area)
        return IoU

    def local_part_attribution(self,target_detection,save_dir):
        bbox = target_detection['bbox']
        for index in tqdm.tqdm(range(self.n_masks)):
            mask,ana_area = self.mask_function(bbox)  
            masked = self.mask_local_image(self.image, mask, ana_area) 
            result = Bayesian_detector.inference(masked,sample_nums=self.sampling_nums)
            result ,_ = Bayesian_detector.nms_results(result ,dict(type='BT_nms', iou_thr=0.6))
            new_detections = []
            for det in result:
                if det['score']>0.05:
                    new_detections.append(det)
            response_of_masked(new_detections,[target_detection],mask,index,save_dir)
        _,ana_area = self.mask_function(bbox)  
        res = np.zeros((ana_area[3]-ana_area[1], ana_area[2]-ana_area[0]), dtype=np.float32)
        res_for_targets = np.array(res)   

        for index in range(self.n_masks):
            mask_path = os.path.join(save_dir,'mask-{0}.npy'.format(str(index)))
            weight_path = os.path.join(save_dir,'weight-{0}.npy'.format(str(index)))
            mask = np.load(mask_path)
            weight = np.load(weight_path)
            for score_for_one_mask in weight:
                if score_for_one_mask < 0:
                    map_single = (mask) * np.abs(score_for_one_mask)
                    res_for_targets = res_for_targets + np.array(map_single) 
        return res_for_targets,ana_area

    def generate_mask_for_one_target(self, bbox):
        if len(bbox) == 8:
            xmin = min(bbox[2*i] for i in range(4))
            ymin = min(bbox[2*i+1] for i in range(4))
            xmax = max(bbox[2*i] for i in range(4))
            ymax = max(bbox[2*i+1] for i in range(4))
            bbox_w = xmax - xmin
            bbox_h = ymax - ymin
            bbox = [xmin,ymin,xmax,ymax]
        else:
            bbox_w = bbox[2] - bbox[0]
            bbox_h = bbox[3] - bbox[1] 

        image_h, image_w = self.image_size 
        area = max(bbox_w,bbox_h)
        ana_area = [max(0,round(bbox[0]-area)),max(0,round(bbox[1]-area)),
                    min(image_w,round(bbox[2]+area)),min(image_h,round(bbox[3]+area))]   

        ana_w = ana_area[2] - ana_area[0]
        ana_h = ana_area[3] - ana_area[1]

        grid_w, grid_h = self.grid_size
        cell_w, cell_h = math.ceil(ana_w / grid_w), math.ceil(ana_h / grid_h)
        up_w, up_h = (grid_w + 1) * cell_w, (grid_h + 1) * cell_h
        mask = (np.random.uniform(0, 1, size=(grid_h, grid_w)) <
                self.prob_thresh).astype(np.float32)
        mask = cv2.resize(mask, (up_w, up_h), interpolation=cv2.INTER_LINEAR)
        offset_w = np.random.randint(0, cell_w)
        offset_h = np.random.randint(0, cell_h)
        mask = mask[offset_h:offset_h + ana_h, offset_w:offset_w + ana_w]   
        return mask, ana_area

    def mask_local_image(self,image,mask,ana_area):
        whole_mask = np.ones(image.shape[:2])
        whole_mask[ana_area[1]:ana_area[3],ana_area[0]:ana_area[2]] = mask
        masked = ((image.astype(np.float32) / 255 * np.dstack([whole_mask] * 3)) *
                255).astype(np.uint8)
        return masked
    
    def spatial_quality(self,target_instance,det_instance):
        gt_box = target_instance.bounding_box
        gt_seg_mat = target_instance.segmentation_mask
        bg_seg_mat = np.ones(AS.image_size, dtype=np.bool)
        bg_seg_mat[int(gt_box[1]):int(gt_box[3])+1, int(gt_box[0]):int(gt_box[2])+1] = False
        num_fg_pixels_vec = target_instance.num_pixels
        
        det_seg_heatmap_mat = det_instance.calc_heatmap(AS.image_size)
        fg_loss = _calc_fg_loss(np.stack([gt_seg_mat],axis=2), np.stack([det_seg_heatmap_mat],axis=2))
        bg_loss = _calc_bg_loss(np.stack([bg_seg_mat],axis=2), np.stack([det_seg_heatmap_mat],axis=2))
        spatial_qual = _calc_spatial_qual(fg_loss, bg_loss, np.array([[num_fg_pixels_vec]])) 
        return spatial_qual

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Bayesian inference')
    parser.add_argument('--img', default='demo/000009.jpg',help='image path')
    parser.add_argument('--config', default='ckpt/SSDD+/FCOS_MCdropout/fcos_obb_r50_caffe_fpn_gn-head_4x4_SSDD+_MCdropout.py',help='test config file path')
    parser.add_argument('--checkpoint',default='ckpt/SSDD+/FCOS_MCdropout/epoch_36.pth',help='checkpoint path')
    parser.add_argument('--device', type=str, default=0, help="set gpu id for attribution") 
    parser.add_argument('--save_dir', type=str, default=None, help="save_dir, default to be saved in the result path") 
    parser.add_argument('--result', default='ckpt/SSDD+/FCOS_MCdropout/out/000009/prob_results.txt' ,help='the result needed to be analysed')
    parser.add_argument('--type', default='local' ,help='attribution type: only local')
    args = parser.parse_args()

    image_path = args.img
    image_name = image_path.split('/')[-1].split('.')[0]
    config = args.config
    checkpoint = args.checkpoint
    device = "cuda:{0}".format(args.device)
    attribution_type = args.type

    ## build the detector ##
    model = init_detector(config, checkpoint, device)
    model.apply(apply_dropout)
    Bayesian_detector = Approximate_Bayesian_Detector(model)

    ## prepare the image ##
    image = cv2.imread(image_path)

    ## prepare the result ##
    result = read_result(args.result)
    ## prepare the save dir ##
    save_dir = args.save_dir if args.save_dir else os.path.dirname(args.result)

    AS = attribution_analysis(image, result, attribution_type, save_dir, n_masks=2000)

    if attribution_type=='local':
        index = 0
        for target_detection in AS.targets:
            mask_and_weight_path = os.path.join(AS.save_dir,'{0}/mask_and_weight'.format(str(index)))
            if not os.path.exists(mask_and_weight_path):
                os.makedirs(mask_and_weight_path)
            saliency_map,ana_area = AS.local_part_attribution(target_detection,mask_and_weight_path)

            image_with_bbox = image.copy()
            bbox = [int(i) for i in target_detection['bbox']]
            if len(bbox) == 4:
                cv2.rectangle(image_with_bbox, tuple(bbox[:2]), tuple(bbox[2:]), (0, 255, 0), 3)
            else:
                pt = np.array([[bbox[2*i],bbox[2*i+1]] for i in range(4)])
                cv2.polylines(image_with_bbox,[pt],True,(0, 255, 0),3)
            plt.figure(figsize=(7, 7),dpi=300)
            plt.imshow(image_with_bbox[ana_area[1]:ana_area[3],ana_area[0]:ana_area[2], ::-1])
            plt.imshow(saliency_map, cmap='jet', alpha=0.5)
            plt.axis('off')
            plt.savefig(os.path.join(os.path.join(AS.save_dir,str(index)),'saliency-{0}.jpg'.format(str(index))))
            plt.cla()
            plt.clf()
            plt.close()
            np.save(os.path.join(os.path.join(AS.save_dir,str(index)),'saliency-{0}.npy'.format(str(index))),saliency_map)
            index += 1