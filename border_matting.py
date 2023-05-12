import numpy as np
import cv2
import math

'''
Implementation of the border matting algorithm based on paper:
Available at: https://dl.acm.org/doi/10.1145/1186562.1015720?fbclid=IwAR2gI_CV9mQmJD8lLp_DoHJw9Zl8QdKWxUp_J7CFLr_god47yYVqp1w_-TE

This is the term project of digital image processing course in NTU.
This code do border matting on a given image and its trimap.

Input: a preprocessed image's trimap
Output: a resulting alpha map after applying border matting

Note: imcomplete code, still debugging

'''

class BorderMatting:
    def __init__(self, img, trimap):
        self.img = img
        self.trimap = trimap                        # 0: background, 4: foreground
        self.w = 6                                  # for constructing new trimap
        self.lambda1 = 50                           # for smoothing regularizer
        self.lambda2 = 1000                         # for smoothing regularizer
        self.L = 20                                 # for sampling mean and sample variance (41-1)/2
        self.delta_level = 30                       # for minimizing energy function (DP)
        self.sigma_level = 10                       # for minimizing energy function (DP)
        self.C = []                                 # contour
        self.D = dict()                             # dictionary for t(n): format (xt, yt): [(x1, y1), ...]
        self.delta_sigma_dict = dict()              # dictionary for delta and sigma: format (xt, yt): (delta, sigma)

    def run(self):
        self.find_coutour()                         # find contour
        self.pixel_group()                          # group pixels and map them to contour pixels
        self.energy_function()                      # minimizing energy function: find delta and sigma pairs
        alpha_map = self.construct_alpha_map()      # use best delta and sigma pairs to construct alpha map
        return alpha_map


    ''' Main Utility Functions '''

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
    
    def pixel_group(self):
        # find nearest contour pixel for each pixel in trimap
        for point in self.C:
            self.D[point] = []
        
        m, n = self.trimap.shape
        for i in range(m):
            for j in range(n):
                min_dist = 100000000
                min_point = None
                for point in self.C:
                    dist = (i - point[0]) ** 2 + (j - point[1]) ** 2
                    if dist < min_dist:
                        min_dist = dist
                        min_point = point
                if min_dist < self.w ** 2:
                    self.D[min_point].append((i, j))
        return
    

    def energy_function(self):
        ''' equation (12) in the paper '''
        # TODO: check errors
        # previous delta and sigma
        _delta = self.delta_level
        _sigma = self.sigma_level

        for point in self.C:
            energy = 10000000000000
            best_delta = None
            best_sigma = None
            for delta in range(self.delta_level):
                for sigma in range(self.sigma_level):
                    V = self.smoothing_regularizer(delta, _delta, sigma, _sigma)
                    D = 0
                    pixel_group = self.D[point]
                    for pixel in pixel_group:
                        distance = (pixel[0] - point[0]) ** 2 + (pixel[1] - point[1]) ** 2
                        alpha = self.distance_to_alpha(distance, sigma, delta)
                        D += self.data_term(alpha)
                    if energy > V + D:
                        energy = V + D
                        best_delta = delta
                        best_sigma = sigma
            self.delta_sigma_dict[point] = (best_delta, best_sigma)
        return


    def construct_alpha_map(self):
        alpha_map = np.copy(self.trimap)
        for point in self.C:
            delta, sigma = self.delta_sigma_dict[point]
            pixel_group = self.D[point]
            for pixel in pixel_group:
                distance = (pixel[0] - point[0]) ** 2 + (pixel[1] - point[1]) ** 2
                alpha = self.distance_to_alpha(distance, sigma, delta)
                alpha_map[pixel] = alpha
        return alpha_map
    

    ''' Important equations in the paper '''

    def smoothing_regularizer(self, m1, v1, m2, v2):
        ''' equation (13) in the paper '''
        return self.lambda1 * (m1 - m2) ** 2 + self.lambda2 * (v1 - v2) ** 2


    def data_term(self, alpha):
        ''' equation (14) in the paper '''
        # TODO: log2 or log10?
        return -1 * math.log(self.gaussian(alpha, self.alpha_mean(alpha), self.alpha_variance(alpha)))
    

    def alpha_mean(self, alpha, pos):
        ''' equation (15) in the paper '''
        return (1 - alpha) * self.sample_mean(pos, 0) + alpha * self.sample_mean(pos, 1)
    

    def alpha_variance(self, alpha, pos):
        ''' equation (15) in the paper '''
        return (1 - alpha) ** 2 * self.sample_variance(pos, 0) + alpha ** 2 * self.sample_variance(pos, 1)
    

    ''' Helper Functions '''

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
    

    def distance_to_alpha(self, distance, sigma, delta):
        return 1 / (1 + np.exp(-1 * (distance - delta) / sigma))
    

    def gaussian(self, x, mean, variance):
        return np.exp(-(x - mean) ** 2 / (2 * variance)) / np.sqrt(2 * np.pi * variance)
