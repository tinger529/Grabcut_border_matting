# Grabcut_border_matting
**Implementation of grabcut border matting**

Implementation of the border matting algorithm based on paper:
Available at: https://dl.acm.org/doi/10.1145/1186562.1015720?fbclid=IwAR2gI_CV9mQmJD8lLp_DoHJw9Zl8QdKWxUp_J7CFLr_god47yYVqp1w_-TE

This is the term project of digital image processing course in NTU.
This code do border matting on a given image and its trimap.

Input: a preprocessed image's trimap
Output: a resulting alpha map after applying border matting

There are five main steps:

1. First we need to find the contour on the given trimap, i.e. the border pixels of TF and TB. I use dilation and canny to find the edge that is a little bit outside the border, to make sure the alpha values of foreground pixels will not change too much.
2. We need to find TU area and match each pixel on this area to its closet contour pixel. I make it a dictionary, the keys are the contour pixels and the values are pixels in TU.
3. Then we have to define two important equations, according to the paper, energy function and smoothing regularizer.
4. Start minimizing energy function using DP. Find the best sigma and delta for each pixel on the contour.
5. After calculating sigma and delta for each pixel on the contour, contruct the resulting alpha map.

Remain problems:

1. If use the same settings as the paper provides, it takes too much time to minimize the energy.
