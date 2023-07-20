#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
KorniaSLAM using some OpenCV functionalities

Author: Zeeshan Khan Suri (zshn25.github.io)
License: MIT
"""


from collections import defaultdict
from functools import partial
import glob

from matplotlib import pyplot as plt
import numpy as np
import cv2

from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform

import torch
from torch.nn import functional as F

import torchvision
from torchvision.datasets.video_utils import VideoClips

import kornia
import kornia.feature as KF
from kornia_moons.feature import *
import kornia.geometry as KG


class Model():
    def __init__(self):
        # Initiate ORB detector
        features = cv2.SIFT_create()
        self.feature_extractor = partial(features.detectAndCompute, mask=None)

        # Store previous frames' data
        self.last = defaultdict()

        # Feature matcher
        # FLANN parameters for SIFT, SURF
        # FLANN_INDEX_KDTREE = 1
        # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        # search_params = dict(checks=50)   # or pass empty dictionary
        # self.flann = cv2.FlannBasedMatcher(index_params,search_params)
        # self.matcher = partial(self.flann.knnMatch, k=2)
        bfmatcher = cv2.BFMatcher()
        self.matcher = partial(bfmatcher.knnMatch, k=2)

        # Kornia
        features = OpenCVDetectorKornia(cv2.SIFT_create())
        # features = KF.ScaleSpaceDetector(resp_module=KF.CornerGFTT())
        self.local_feature = KF.LocalFeature(
            features, KF.LAFDescriptor(KF.MKDDescriptor()), )
        # self.local_feature = KF.SIFTFeature(num_features=1000, upright=True, device=device)

        # Matches Filter
        self.ransac_filter = partial(ransac,
                                     model_class=FundamentalMatrixTransform,
                                     min_samples=8,
                                     residual_threshold=1,
                                     max_trials=100)

    def __call__(self, frame):
        # find the keypoints and descriptors with ORB
        # H,W,_ = frame.shape
        # frame = cv2.resize(frame, (W//2, H//2))
        kps1, des1 = self.feature_extractor(frame)
        matches = None

        if self.last:
            # Do matching with previous frames' data
            kps2, des2 = self.last["kp"], self.last["des"]
            frame2 = self.last["frame"]
            matches = self.matcher(des1, des2)
            filtered_matches, model = self.filter_matches(matches, kps1, kps2)

            # Extract rotation and translation from Essential matrix
            R, t = self.get_Rt_fromE(model)

            # Draw
            draw_opencv(filtered_matches, frame)

        self.last.update({"kp": kps1, "des": des1, "frame": frame})
        return matches

    def filter_matches(self, matches, kps1, kps2):
        # ratio test as per Lowe's paper
        ret = [(kps1[m.queryIdx].pt, kps2[m.trainIdx].pt)
               for m, n in matches if m.distance < 0.8*n.distance]

        # Filter based on fundemental matrix
        if len(ret) > 0:
            ret = np.array(ret).astype(int)
            # F, inliers = cv2.findFundamentalMat(np.int32(ret[:,0]), np.int32(ret[:,1]), cv2.FM_LMEDS)
            model, inliers = self.ransac_filter((ret[:, 0], ret[:, 1]))

        return ret[inliers], model

    def get_Rt_fromE(self, model):
        W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        u, w, vt = np.linalg.svd(model.params)

        assert np.linalg.det(u) > 0

        if np.linalg.det(vt) < 0:
            vt = -vt

        R = u @ W @ vt
        if (R.diagonal()).sum() < 0:
            R = u @ W.T @ vt
        t = u[:, 2]

        return R, t

    @torch.inference_mode()
    def sift_korniadesc_matching(self, timg1):

        def get_matching_kpts(lafs1, lafs2, idxs):
            src_pts = KF.get_laf_center(
                lafs1).view(-1, 2)[idxs[:, 0]].detach().cpu().numpy().astype(int)
            dst_pts = KF.get_laf_center(
                lafs2).view(-1, 2)[idxs[:, 1]].detach().cpu().numpy().astype(int)
            return src_pts, dst_pts

        lafs1, resps1, descs1 = self.local_feature(timg1)
        if self.last:
            lafs2, resps2, descs2 = self.last["laf"], self.last["resp"], self.last["desc"]
            dists, idxs = KF.match_adalam(
                descs1[0], descs2[0], lafs1, lafs2, hw1=timg1.shape[2:], hw2=timg1.shape[2:])

            src_pts, dst_pts = get_matching_kpts(lafs1, lafs2, idxs)

            if len(src_pts) < 2:
                return

            model, inliers_mask = self.ransac_filter((src_pts, dst_pts))

            # Extract rotation and translation from Essential matrix
            R, t = self.get_Rt_fromE(model)

            if inliers_mask.sum() > 2:

                filtered_matches = np.concatenate([src_pts[inliers_mask.squeeze().astype(bool), :][:, np.newaxis, :],
                                                   dst_pts[inliers_mask.squeeze().astype(bool), :][:, np.newaxis, :]], axis=1)

                # Draw
                draw_opencv(filtered_matches, torch_to_opencv(timg1))

            # Save filtered features to track them
        #     self.last.update({"laf": lafs1[:, idxs[:, 0][inliers_mask], ...],
        #                       "resp": resps1[:, idxs[:, 0][inliers_mask], ...],
        #                       "desc": descs1[:, idxs[:, 0][inliers_mask], ...]})
        # else:
        self.last.update({"laf": lafs1,
                          "resp": resps1,
                          "desc": descs1})


def torch_to_opencv(frame):
    """Convert torch.Tensor to OpenCV Image"""
    assert frame.dim() == 4 and frame.shape[1] == 3
    frame = kornia.color.bgr_to_rgb(frame)
    frame_opencv = kornia.tensor_to_image(frame)*255
    return frame_opencv.astype(np.uint8).copy()


def draw_opencv(matches, frame):
    """Draw using OpenCV for visualization"""
    for (u1, v1), (u2, v2) in matches:
        cv2.circle(frame, (u1, v1), color=(0, 255, 0), radius=3)
        cv2.line(frame, (u1, v1), (u2, v2), color=(255, 0, 0))
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == ord('q'):
        exit()


if __name__ == "__main__":
    # Test

    # reader = torchvision.io.VideoReader("/home/z/data/test.mp4")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = "cpu"
    kitti = glob.glob(
        "/home/z/data/kitti_data/2011_09_30/2011_09_30_drive_0028_sync/image_02/data/*.jpg")
    kitti.sort()
    # K = np.array([])
    # Initiate model
    model = Model()
    fig, ax = plt.subplots()
    # for i, data in enumerate(reader):
    for im_path in kitti:
        frame = torchvision.io.read_image(
            im_path).detach().to(device).unsqueeze(0).float()
        frame /= 255.
        _, _, H, W = frame.shape
        frame = torchvision.transforms.Resize((192, 640))(frame).to(device)

        # OpenCV
        # cv_frame = torch_to_opencv(frame)
        # processed = model(cv_frame)

        # Kornia
        model.sift_korniadesc_matching(frame)

    cv2.destroyAllWindows()
