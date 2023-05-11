import numpy as np
import cv2

'''
Implementation of the border matting algorithm based on paper:
Available at: https://dl.acm.org/doi/10.1145/1186562.1015720?fbclid=IwAR2gI_CV9mQmJD8lLp_DoHJw9Zl8QdKWxUp_J7CFLr_god47yYVqp1w_-TE

This is the term project of digital image processing course in NTU.
This code do border matting on a given image and its trimap.

Input: a preprocessed image's trimap
Output: a resulting alpha map after applying border matting

Note: imcomplete code, not working yet

'''

class BorderMatting:
    def __init__(self, img, trimap):
        self.img = img
        self.trimap = trimap                    # 0: background, 4: foreground
        self.w = 6                              # for constructing new trimap
        self.lambda1 = 50                       # for smoothing regularizer
        self.lambda2 = 1000                     # for smoothing regularizer
        self.L = 20                             # for sampling mean and sample variance (41-1)/2
        self.C = []                             # contour


    def find_coutour(self):
        # TODO: use cv2 or hand-crafted one
        edges = cv2.Canny(self.trimap, threshold1=2, threshold2=3)

        # construct new trimap
        newmap = np.zeros_like(self.trimap)
        newmap[edges == 1] = 2
        newmap[self.trimap == 0] = 0
        newmap[self.trimap == 4] = 4
        self.trimap = newmap

        # find contour
        indices = np.where(edges == 1)
        self.C = list(zip(indices[0], indices[1]))   
        return 
    

    def energy_function(self, alphas, params):
        # TODO: minimize the energy function
        raise NotImplementedError


    def smoothing_regularizer(self, m1, v1, m2, v2):
        ''' equation (13) in the paper '''
        return self.lambda1 * (m1 - m2) ** 2 + self.lambda2 * (v1 - v2) ** 2


    def data_term(self, alpha):
        # TODO: compute the data term
        raise NotImplementedError


    def sample_mean(self, pos, alpha):
        area = self.trimap[pos[0] - self.L: pos[0] + self.L + 1, pos[1] - self.L: pos[1] + self.L + 1]
        if alpha == 0:  # background
            mean = np.sum(area[area == 0]) / self.L ** 2
        else:           # foreground
            mean = np.sum(area[area == 1]) / self.L ** 2
        return mean
    

    def sample_variance(self, pos, alpha):
        area = self.trimap[pos[0] - self.L: pos[0] + self.L + 1, pos[1] - self.L: pos[1] + self.L + 1]
        if alpha == 0:  # background
            variance = np.sum((area[area == 0] - self.sample_mean(pos, alpha)) ** 2) / self.L ** 2
        else:           # foreground
            variance = np.sum((area[area == 1] - self.sample_mean(pos, alpha)) ** 2) / self.L ** 2
        return variance
    

    def alpha_mean(self, alpha, pos):
        ''' equation (15) in the paper '''
        return (1 - alpha) * self.sample_mean(pos, 0) + alpha * self.sample_mean(pos, 1)
    

    def alpha_variance(self, alpha, pos):
        ''' equation (15) in the paper '''
        return (1 - alpha) ** 2 * self.sample_variance(pos, 0) + alpha ** 2 * self.sample_variance(pos, 1)
