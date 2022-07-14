# Jerret Stovall
# CS 3150
# Final Project
# Dr. Feng

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def find_local_min(hist):
    kern = np.array(
        [2, 0, 0, 0,
         2, 0, 0, 0,
         2, 0, 0, 0,
         2, 0, 0, 0,
         1, 0, 0, 0,
         1, 0, 0, 0,
         1, 0, 0, 0,
         1, 0, 0, 0,
         -3, -3, -3, -3
         - 3, -3, -3, -3
            , 0, 0, 0, 1
            , 0, 0, 0, 1
            , 0, 0, 0, 1
            , 0, 0, 0, 1
            , 0, 0, 0, 2
            , 0, 0, 0, 2
            , 0, 0, 0, 2
            , 0, 0, 0, 2])

    hist[0] = 0
    deriv_value = np.convolve(hist, kern, mode='same')
    threshold_value = deriv_value.argmax()
    return threshold_value, deriv_value


# Read in image to be scanned
src_img = cv.imread('./earhats.jpg')
plt.figure()
plt.title('Source Image')
plt.imshow(cv.cvtColor(src_img, cv.COLOR_BGR2RGB))

# Perform gamma correction
# Convert to LUV
luv_img = cv.cvtColor(src_img, cv.COLOR_BGR2LUV)
l = luv_img[:, :, 0]
u = luv_img[:, :, 1]
v = luv_img[:, :, 2]

# Create histograms
histogram, bins = np.histogram(l.ravel(), 256, [0, 256])
thresh_value, dx = find_local_min(histogram)

# Perform Gamma Correction
l = l / 256.0
l = l ** 0.6
l = l * 256
luv_img[:, :, 0] = l
gamma_img = cv.cvtColor(luv_img, cv.COLOR_LUV2BGR)

# Convert source image to gray scale
gry_img = cv.cvtColor(gamma_img, cv.COLOR_BGR2GRAY)
plt.figure()
plt.title('Gray Scale of Source Image With Gamma Correction')
plt.imshow(gry_img, cmap='gray')

# Apply a gaussian blur with a kernal size of 7x7
gry_img = cv.GaussianBlur(gry_img, (13,13), cv.BORDER_DEFAULT)
plt.figure()
plt.title('Gray Scale of Source Image With Gaussian Blur')
plt.imshow(gry_img, cmap='gray')

# Detect circles using built in Hough Transform Method
circles = cv.HoughCircles(gry_img, cv.HOUGH_GRADIENT, 1, 150, param1=50, param2=30, minRadius=60, maxRadius=0)

det_circles = np.uint16(np.around(circles))

for i in det_circles[0,:]:
    cv.circle(src_img, (i[0],i[1]), i[2], (0,255,0), 5)

plt.figure()
plt.title('Ears Detected')
plt.imshow(cv.cvtColor(src_img, cv.COLOR_BGR2RGB))

plt.show()
