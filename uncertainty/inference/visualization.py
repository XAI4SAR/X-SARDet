import numpy as np
from matplotlib.patches import Ellipse
import BboxToolkit as bt

class detection_visualization():
    def __init__(self,ax,image_data,mode='ellipse'):
        self.ax = ax 
        self.mode = mode
        self.image_data = image_data
        self.img_size = (image_data.shape[0],image_data.shape[1])

    def plot_detections(self, bbox, score=None, cls_uncertainty=None, covs=None, color='b'):
        if cls_uncertainty:
            score = str(round(score*100,3)) + '%'
            cls_uncertainty = str(round(cls_uncertainty*100,3)) + '%'  
            text = 'S:'+score+'  U:'+cls_uncertainty
        elif score:
            score = str(round(score*100,3)) + '%'
            text = 'S:'+score
        else:
            text = None
        if len(bbox)==4:
            bt.draw_hbb(self.ax,np.array([bbox]),[text],color,font_size=10)
        else:
            bt.draw_poly(self.ax,np.array([bbox]),[text],color,font_size=10) 
        if covs:
            self.plot_reg_unertainty(bbox,covs,color)

    def plot_reg_unertainty(self,bbox,covs,color):
        if self.mode == 'arrow' or self.mode == 'ellipse':
            self.draw_cov(self.ax,bbox,covs,color,self.mode)

    def draw_cov(self, ax, box, covs, colour='b', mode='ellipse'):
        if mode == 'arrow' or mode == 'ellipse':
            for i in range(len(covs)):
                cov = covs[i]
                # Calculate the eigenvalues and eigenvectors for each corner covariances
                vals, vecs = np.linalg.eig(cov)
                # flip the vectors along the y axis (for plotting as y is inverted)
                vecs[:, 1] *= -1
                # Determine which eigenvalue/vector is largest and which covariance that corresponds to
                val_max_idx = np.argmax(vals)
                argmax_cov = np.argmax(np.diag(cov))
                # Calculate the magnitudes along each eigenvector used for visualization (2 * std dev)
                magnitude_1 = np.sqrt(cov[argmax_cov][argmax_cov]) * 2
                magnitude_2 = np.sqrt(covs[0][np.abs(1 - argmax_cov)][np.abs(1 - argmax_cov)])*2
                # Calculate the end-points of the eigenvector with given magnitude
                end1 = vecs[val_max_idx] * magnitude_1 + np.array([box[2*i], box[2*i+1]])
                end2 = vecs[np.abs(1 - val_max_idx)] * magnitude_2 + np.array([box[2*i], box[2*i+1]])
                # Calculate the difference in the x and y direction of the corners
                dx1 = end1[0] - box[2*i]
                dy1 = end1[1] - box[2*i+1]
                dx2 = end2[0] - box[2*i]
                dy2 = end2[1] - box[2*i+1]   
                # Draw the appropriate corner representation of Gaussian Corners    
                if mode == 'arrow':  
                   # Draw the arrows only if they have magnitude
                    if dx1 + dy1 != 0:
                        ax.arrow(box[2*i], box[2*i+1], dx1, dy1, width=1, color=colour,
                                length_includes_head=True, head_width=6, head_length=2)
                    if dx2 + dy2 != 0:
                        ax.arrow(box[2*i], box[2*i+1], dx2, dy2, width=1, color=colour,
                                length_includes_head=True, head_width=6, head_length=2)
                if mode == 'ellipse':
                    self.draw_ellipse_corner(ax, (box[2*i], box[2*i+1]), width1=magnitude_2, height1=magnitude_1,
                                        dx=dx1, dy=dy1, colour=colour)                  

    def draw_ellipse_corner(self, ax, centre, width1, height1, dx, dy, colour='b'):
        # Calculate the angle to tilt the ellipses (largest eigenvector set to height)
        if dx == 0:
            tl_angle = 0
        elif dy == 0:
            tl_angle = 90
        else:
            tl_angle = 90 + np.arctan(dy / dx) * (180 / np.pi)

        ax.add_patch(Ellipse(centre, width=10*width1, height=10*height1, angle=tl_angle,
                            linewidth=1, edgecolor=colour, facecolor=None, fill=False, color='lime'))


