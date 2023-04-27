import argparse
from mmdet.apis import init_detector, inference_detector
import torch.nn as nn
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import BboxToolkit as bt
from shapely.geometry import Polygon
from uncertainty.inference.visualization import detection_visualization
from matplotlib.patches import Polygon as P
from mmdet.ops import RoIAlign, RoIPool, nms, nms_rotated


def apply_dropout(m):
    if isinstance(m, nn.Dropout):
        print(m)
        print(m.training)
        m.train()
        print(m.training)

class Approximate_Bayesian_Detector():
    def __init__(self, model):
        self.model = model
        self.iou_threshold = 0.7
    
    def iou_poly(self,bboxes):
        area1 = bboxes[..., 2] * bboxes[..., 3]
        bboxes = bt.obb2poly(bboxes)
        target_bbox = bboxes[0]
        target_bbox = [(target_bbox[2*i],target_bbox[2*i+1]) for i in range(int(len(target_bbox)/2))]
        iter_areas = []
        for bbox in bboxes[1:]:
            bbox = [(bbox[2*i],bbox[2*i+1]) for i in range(int(len(bbox)/2))]
            ploy_A = Polygon(target_bbox).convex_hull
            ploy_B = Polygon(bbox).convex_hull

            if not ploy_A.intersects(ploy_B):
                iter_area = 0.0
            else: 
                iter_area = ploy_A.intersection(ploy_B).area    
            iter_areas.append(iter_area)   
        area2 =  np.array(iter_areas)
        iou_score = area2/(area1[0]+area1[1:]-area2)
        return iou_score

    def iou_hbb(self,bboxes):
        area1 = (bboxes[:,2] - bboxes[:,0])*(bboxes[:,3]-bboxes[:,1])
        xx1 = np.maximum(bboxes[1:,0],bboxes[0][0])
        yy1 = np.maximum(bboxes[1:,1],bboxes[0][1])
        xx2 = np.minimum(bboxes[1:,2],bboxes[0][2])
        yy2 = np.minimum(bboxes[1:,3],bboxes[0][3])
        area2 = np.maximum(xx2-xx1,0)*np.maximum(yy2-yy1,0)
        iou_score = area2/(area1[0]+area1[1:]-area2)
        return iou_score
    
    def get_observation_by_iou(self, bboxes):
        bboxes = np.array(bboxes.tolist())
        if bboxes.shape[1] == 4: # for HBB
            iou_score = self.iou_hbb(bboxes)
        elif bboxes.shape[1] == 5:
            iou_score = self.iou_poly(bboxes)
        return np.where(iou_score>self.iou_threshold),np.where(iou_score<=self.iou_threshold)

    def UQ_for_classification(self, classifications_scores):
        std = np.std(classifications_scores)
        return std


    def UQ_for_localization(self, bboxs, final_bbox):
        """
        for ploy, get the w and h of Minimum circumscribed rectangle
        """
        std = np.std(bboxs,0) # [std_x,std_y,...]

        if len(final_bbox) == 4:
            w = final_bbox[2] - final_bbox[0]
            h = final_bbox[3] - final_bbox[1]
            for i in range(2):
                std[2*i] = std[2*i] / w
                std[2*i+1] = std[2*i+1] / h    

        elif len(final_bbox) == 8:
            xmin = min(final_bbox[2*i] for i in range(4))
            xmax = max(final_bbox[2*i] for i in range(4))
            ymin = min(final_bbox[2*i+1] for i in range(4))
            ymax = max(final_bbox[2*i+1] for i in range(4))
            w = xmax - xmin
            h = ymax - ymin
            for i in range(4):
                std[2*i] = std[2*i] / w
                std[2*i+1] = std[2*i+1] / h
        else:
            print("5 paramenter is not supported")
        return std

    def MC_dropout_inference(self,image,sampling_nums=10):
        self.sampling_nums = sampling_nums
        self.merge_multi_forward_results = []
        for i in range(self.sampling_nums):
            out = inference_detector(self.model, image)
            for class_id, pred in enumerate(out):
                for *box, score in pred:
                    classification_vector = [0] *(len(out)+1) # the last one is for background
                    classification_vector[class_id] = score
                    one_detection = [box, classification_vector]
                    self.merge_multi_forward_results.append(one_detection)
        return self.merge_multi_forward_results


    def observations_reconstruction(self,merge_multi_forward_results,sampling_nums):
        # calculate iou among bboxs and produce observations
        self.observations = {}
        num_of_observations = 0
        results = np.array(merge_multi_forward_results,dtype=object)
        while results.shape[0] > 0:
            num_of_observations += 1
            bboxes = results[:,0]
            keep,the_rest = self.get_observation_by_iou(bboxes)
            one_observation = [results[0].tolist()]
            results = results[1:]
            bboxs_appro = results[keep]
            bboxs_appro = bboxs_appro.tolist()
            one_observation += bboxs_appro
            while len(one_observation) < sampling_nums:
                one_observation.append([[],[0,1]]) # only for single class
            self.observations[str(num_of_observations)] = one_observation
            if results.shape[0] != 0:
                if the_rest[0].shape[0] != 0:
                    results = results[the_rest]
                else:
                    results = np.array([])
        return self.observations
   
    def change(self,bbox):
        """
        only design for OBB
        """
        return np.append(bbox[4:8],bbox[0:4])

    def results_accumulation(self,observations):
        detections = []
        for observation_id in observations.keys():
            one_observation = np.array(observations[observation_id])
            scores = np.array(one_observation[:,1].tolist())
            scores = np.array(scores[:,0].tolist()) # only design for single class
            bboxes = one_observation[:,0].tolist()
            bboxes = np.array([bbox for bbox in bboxes if bbox])

            if bboxes.shape[1] == 5:
                index = np.where(bboxes[:,-1]<0)
                bboxes = bt.obb2poly(bboxes)
                for i in index[0].tolist():
                    bboxes[i] = self.change(bboxes[i])

            final_observed_bbox = np.mean(bboxes, 0).tolist()
            final_score = np.mean(scores)
            uncertainty_of_classfication = self.UQ_for_classification(scores)
            uncertainty_of_localization = self.UQ_for_localization(bboxes,final_observed_bbox).tolist()
            covs = self.get_covs_of_bboxes(bboxes)
            one_detection = dict()
            one_detection['score'] = final_score
            one_detection['bbox'] = final_observed_bbox
            one_detection['cls-uncertainty'] = uncertainty_of_classfication
            one_detection['reg-uncertainty'] = uncertainty_of_localization
            one_detection['reg-covs'] = covs
            detections.append(one_detection)
        self.out = np.array(detections)
        return self.out
    
    def obb2hbb(self,box):
        if len(box) == 4:
            return box
        elif len(box) == 8:
            xmin = min(box[2*i] for i in range(4))
            xmax = max(box[2*i] for i in range(4))
            ymin = min(box[2*i+1] for i in range(4))
            ymax = max(box[2*i+1] for i in range(4))
            return [xmin,ymin,xmax,ymax]
        
    def get_covs_of_bboxes(self,bboxes):

        if bboxes.shape[1] == 5:
            bboxes = bt.obb2poly(bboxes)

            new_bbox = []
            for box in bboxes:
                new_bbox.append(self.obb2hbb(box))
            bboxes = np.array(new_bbox)
        
        points = []
        for i in range(int(bboxes.shape[1]/2)):
            if bboxes.shape[0]==1:
                point = [[1e-14,0],[0,1e-14]]
            else:
                point = np.cov(bboxes[:,2*i],bboxes[:,2*i+1]).tolist()
            points.append(point)
        return points

    def nms_results(self, results, nms_cfg):
        if len(results)>0:
            nms_cfg_ = nms_cfg.copy()
            nms_type = nms_cfg_.pop('type', 'BT_nms')
            try:
                nms_op = getattr(nms_rotated, nms_type)
            except AttributeError:
                nms_op = getattr(nms, nms_type)

            bboxes = np.array([dets['bbox'] for dets in results])
            scores = np.array([dets['score'] for dets in results]).reshape(-1,1)
            cls_result = np.concatenate([bboxes, scores], axis=1)
            _, save_index = nms_op(cls_result, **nms_cfg_)   
            return np.array(results)[save_index].tolist(),save_index
        else:
            return results,[]


    def inference(self, image, sample_nums=10):
        self.image = image
        self.sample_nums = sample_nums
        self.MC_dropout_inference(self.image)
        self.observations_reconstruction(self.merge_multi_forward_results,self.sample_nums)
        out = self.results_accumulation(self.observations)  
        return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bayesian inference')
    parser.add_argument('--img', default='demo/000009.jpg',help='image path')
    parser.add_argument('--config',default='ckpt/SSDD+/FCOS_MCdropout/fcos_obb_r50_caffe_fpn_gn-head_4x4_SSDD+_MCdropout.py', help='test config file path')
    parser.add_argument('--checkpoint',default='ckpt/SSDD+/FCOS_MCdropout/epoch_36.pth', help='checkpoint path')
    parser.add_argument('--out', type=str, default='ckpt/SSDD+/FCOS_MCdropout/out', help='the path for attribution analysis results')
    parser.add_argument('--device', type=str, default=0, help="set gpu id for attribution")
    parser.add_argument('--show', default=True, action="store_true", help='show the results')
    args = parser.parse_args()

    image_path = args.img
    config = args.config
    checkpoint = args.checkpoint
    device = "cuda:{0}".format(args.device)

    ## build the detector ##
    model = init_detector(config, checkpoint, device)
    model.apply(apply_dropout)
    Bayesian_detector = Approximate_Bayesian_Detector(model)

    ## prepare the image ##
    image = cv2.imread(image_path)
    result = Bayesian_detector.inference(image)
    result,save_index = Bayesian_detector.nms_results(result,dict(type='BT_nms', iou_thr=0.6))

    if args.out:
        save_dir = args.out
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        image_name = image_path.split('/')[-1].replace('.jpg','')
        save_dir = os.path.join(save_dir,image_name)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_txt = os.path.join(save_dir,'prob_results.txt')
        f = open(save_txt,'w')
        for det in result:
            if det['score']>0.1:
                f.write(str(det))
                f.write('\n')
        f.close()
    if args.show:
        height,width = image.shape[0:2]
        fig_vis, vis_ax = plt.subplots(figsize=(width/100,height/100),dpi = 400)
        vis_ax.set_axis_off()
        vis_ax.imshow(image)
        vis = detection_visualization(vis_ax,image,mode='ellipse')
        for det in result:
            if det['score']>0.1:
                vis.plot_detections(det['bbox'],det['score'],det['cls-uncertainty'],det['reg-covs'],color='r')
        if args.out:
            plt.savefig(os.path.join(save_dir,'detection-results.jpg'))
        plt.show()



