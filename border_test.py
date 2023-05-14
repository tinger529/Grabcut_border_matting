from border_matting import BorderMatting as bm
import cv2
import numpy as np
import math

'''
Testing border matting algorithm.
'''

img = np.array([[0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255],
                   [0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255],
                   [0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255],
                   [0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255],
                   [0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255]])

cv2.imwrite("testin.png", img)

trimap = np.array([[0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4],
                   [0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4],
                   [0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4],
                   [0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                   [0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]])
test = bm(img, trimap).run()
out = np.zeros_like(img)
m,n = img.shape
for i in range(m):
    for j in range(n):
        out[i][j] = test[i][j] * img[i][j]
cv2.imwrite("testout.png", out)
print(test)