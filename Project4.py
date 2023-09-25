import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

# Load images
img1 = cv2.imread('./set1/1.jpg')
img2 = cv2.imread('./set1/2.jpg')
img3 = cv2.imread('./set1/3.jpg')

##### Intro to homographies #####
height, width = img1.shape[:2]

# rotate 1.jpg clockwise 10 degrees

x = width // 2
y = height // 2
cos = np.cos(np.deg2rad(10))
sin = np.sin(np.deg2rad(10))

rotation_matrix = np.array([
    [cos, -sin, x * (1 - cos) + y * sin],
    [sin, cos, y * (1 - cos) - x * sin],
    [0, 0, 1]], dtype=np.float32)

img1_warped = cv2.warpPerspective(img1, rotation_matrix, (1000, 800))


# translate 2.jpg 100 pixels right
translate = np.array([
    [1, 0, 100],
    [0, 1, 0],
    [0, 0, 1]], dtype=np.float32)

img2_warped = cv2.warpPerspective(img2, translate, (1000, 800))

# shrink 3.jpg by half
shrinking_matrix = np.array([
    [0.5, 0, 0],
    [0, 0.5, 0],
    [0, 0, 1]], dtype=np.float32)
img3_warped = cv2.warpPerspective(img3, shrinking_matrix, (1000, 800))

# plotting transformed images
plt.imshow(cv2.cvtColor(img1_warped, cv2.COLOR_BGR2RGB))
plt.title('Rotated 1.jpg')
plt.show()

plt.imshow(cv2.cvtColor(img2_warped, cv2.COLOR_BGR2RGB))
plt.title('Translated 2.jpg')
plt.show()

plt.imshow(cv2.cvtColor(img3_warped, cv2.COLOR_BGR2RGB))
plt.title('Shrunken 3.jpg')
plt.show()

##### Panoramic Stitching #####

# computing SIFT features
sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
kp3, des3 = sift.detectAndCompute(img3, None)

imgKeyPoints1 = cv2.drawKeypoints(img1, kp1, None)
imgKeyPoints2 = cv2.drawKeypoints(img2, kp2, None)
imgKeyPoints3 = cv2.drawKeypoints(img3, kp3, None)

cv2.imshow('1.jpg', imgKeyPoints1)
cv2.imshow('2.jpg', imgKeyPoints2)
cv2.imshow('3.jpg', imgKeyPoints3)
cv2.waitKey(0)

# matching SIFT features

def top_100_matches(des1, des2):
    dist_mat = distance_matrix(des1, des2)
    top_100_indices = np.argsort(dist_mat, axis=None)[:100]
    top_100 = np.unravel_index(top_100_indices, dist_mat.shape)
    return [(top_100[0][i], top_100[1][i]) for i in range(100)]

best_matches_21 = top_100_matches(des1, des2)
best_matches_23 = top_100_matches(des3, des2)

# visualize matches
def cull_invalid(matches, kp1, kp2):
    valid_matches = []
    for i, j in matches:
        if i < len(kp1) and j < len(kp2):
            valid_matches.append((i, j))
    return valid_matches

valid_matches_21 = cull_invalid(best_matches_21, kp1, kp2)
valid_matches_23 = cull_invalid(best_matches_23, kp3, kp2)

matches_21 = [cv2.DMatch(i, j, 0) for i, j in valid_matches_21]
matches_23 = [cv2.DMatch(i, j, 0) for i, j in valid_matches_23]

connect_21 = cv2.drawMatches(img1, kp1, img2, kp2, matches_21, None, flags=2)
connect_23 = cv2.drawMatches(img3, kp3, img2, kp2, matches_23, None, flags=2)

cv2.imshow('Feature Matches between 1.jpg and 2.jpg', connect_21)
cv2.imshow('Feature Matches between 2.jpg and 3.jpg', connect_23)
cv2.waitKey(0)

# estimating homographies

# RANSAC
from_21 = np.float32([kp1[i].pt for i, j in valid_matches_21])
to_21 = np.float32([kp2[j].pt for i, j in valid_matches_21])
H_21, mask_21 = cv2.findHomography(from_21, to_21, cv2.RANSAC, 2)
from_23 = np.float32([kp3[i].pt for i, j in valid_matches_23])
to_23 = np.float32([kp2[j].pt for i, j in valid_matches_23])
H_23, mask_23 = cv2.findHomography(from_23, to_23, cv2.RANSAC, 2)

# Visualize the inliers after RANSAC
inliers_21 = []
for m, mask_val in zip(matches_21, mask_21.ravel()):
    if mask_val == 1:
        inliers_21.append(m)
inliers_23 = []
for m, mask_val in zip(matches_23, mask_23.ravel()):
    if mask_val == 1:
        inliers_23.append(m)

ransac_21 = cv2.drawMatches(img1, kp1, img2, kp2, inliers_21, None, flags=2)
ransac_23 = cv2.drawMatches(img3, kp3, img2, kp2, inliers_23, None, flags=2)

cv2.imshow('RANSAC Inliers between 1.jpg and 2.jpg', ransac_21)
cv2.imshow('RANSAC Inliers between 2.jpg and 3.jpg', ransac_23)
cv2.waitKey(0)

# 350 pixels to the right, 300 pixels down
translate = np.array([
    [1, 0, 350],
    [0, 1, 300],
    [0, 0, 1]], dtype=np.float32)

# compute homographies
translate_12 = np.dot(translate, H_21)
translate_32 = np.dot(translate, H_23)
translate_2 = translate

img1_warped = cv2.warpPerspective(img1, translate_12, (1000, 800))
img3_warped = cv2.warpPerspective(img3, translate_32, (1000, 800))
img2_warped = cv2.warpPerspective(img2, translate_2, (1000, 800))

# fuse all images
fused = np.maximum(img1_warped, np.maximum(img2_warped, img3_warped))

cv2.imshow('Panoramic Image', fused)
cv2.waitKey(0)