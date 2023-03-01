import cv2
import numpy as np
import solver.descriptor_evaluation as ev
from utils.plt import plot_imgs
import matplotlib.pyplot as plt

def draw_matches(data):
    keypoints1 = [cv2.KeyPoint(int(p[1]), int(p[0]), 1) for p in data['keypoints1']]
    keypoints2 = [cv2.KeyPoint(int(p[1]), int(p[0]), 1) for p in data['keypoints2']]
    inliers = data['inliers'].astype(bool)
    matches = np.array(data['matches'])[inliers].tolist()
    img1 = cv2.merge([data['image1'], data['image1'], data['image1']]) * 255
    img2 = cv2.merge([data['image2'], data['image2'], data['image2']]) * 255
    return cv2.drawMatches(np.uint8(img1), keypoints1, np.uint8(img2), keypoints2, matches,
                           None, matchColor=(0,255,0), singlePointColor=(0, 0, 255),flags=2)


experiments = ['./data/descriptors/hpatches/sp/']

##Check that the image is warped correctly
num_images = 5
for e in experiments:
    orb = True if e[50:55] == 'orb' else False
    outputs = ev.get_homography_matches(e, keep_k_points=1000, correctness_thresh=3, num_images=num_images, orb=orb)
    for output in outputs:
        img1 = output['image1'] * 255
        img2 = output['image2'] * 255
        matched_image = draw_matches(output)
        matched_image = cv2.resize(matched_image,[1280,960])
        cv2.imshow("matched",matched_image)
        cv2.waitKey(0)
        # H = output['homography']
        # warped_img1 = cv2.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))
        # print("img1: {}".format(img1.shape))
        # img1 = np.concatenate([img1, img1, img1], axis=1)
        # warped_img1 = np.stack([warped_img1, warped_img1, warped_img1], axis=2)
        # img2 = np.concatenate([img2, img2, img2], axis=1)
        # plot_imgs([img1 / 255., img2 / 255., warped_img1 / 255.], titles=['img1', 'img2', 'warped_img1'], dpi=200)

##Homography estimation correctness
# for exp in experiments:
#     orb = True if exp[:3] == 'orb' else False
#     correctness = ev.homography_estimation(exp, keep_k_points=1000, correctness_thresh=3, orb=orb)
#     print('> {}: {}'.format(exp, correctness))
