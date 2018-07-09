# -*- coding: utf-8 -*-
import cv2
import numpy as np
import six

import common
# import drawing
from cnn import CnnFPGA

# logging
from logging import getLogger, NullHandler
logger = getLogger(__name__)
logger.addHandler(NullHandler())


def _forward_with_rects(model, img_org, rects, batchsize):
    # Crop and normalize
    cropped_imgs = list()
    for x, y, w, h in rects:
        img = img_org[int(y):int(y + h + 1), int(x):int(x + w + 1), :]
        img = cv2.resize(img, (227, 227))
        img = cv2.normalize(img, None, -0.5, 0.5, cv2.NORM_MINMAX)
        cropped_imgs.append(img)

    detections = list()
    landmarks = list()
    visibilities = list()
    poses = list()
    genders = list()

    # Forward each batch
    for i in six.moves.xrange(0, len(cropped_imgs), batchsize):
        logger.info('Proposed region progress : ' + str( int( (i/len(cropped_imgs))*100) ) + ' %')

        # Create batch
        batch = cropped_imgs[i:i + batchsize]
        # Forward

        y = model(batch)

        for b in range(len(y)):
            detections.append(y[b]['detection'])
            landmarks.append(y[b]['landmark'])
            visibilities.append(y[b]['visibility'])
            poses.append(y[b]['pose'])
            genders.append(y[b]['gender'])

    # Denormalize landmarks
    for i, (x, y, w, h) in enumerate(rects):
        landmarks[i] = landmarks[i].reshape(21, 2)  # (21, 2)
        landmark_offset = np.array([x + w / 2, y + h / 2], dtype=np.float32)
        landmark_denom = np.array([w, h], dtype=np.float32)
        landmarks[i] = landmarks[i] * landmark_denom + landmark_offset

    return detections, landmarks, visibilities, poses, genders

def shit2chocolate(masked_pts):
    masked_pts2 = [ [ i ] for i in masked_pts ]
    return np.array( masked_pts2, dtype=np.float32)

def _proposal_region(pts, pts_mask, img_rect, landmark_pad_rate=0.1):
    # TODO Improve for rotated rectangles
    # AFLW Template
    tpl_pts = np.array([ [[-0.479962468147, 0.471864163876]],
                     [[-0.30303606391, 0.508996844292]],
                     [[-0.106451146305, 0.498075485229]],
                     [[0.106451146305, 0.498075485229]],
                     [[0.30303606391, 0.508996844292]],
                     [[0.479962468147, 0.471864163876]],
                     [[-0.447198301554, 0.321149080992]],
                     [[-0.318325966597, 0.325517624617]],
                     [[-0.163242310286, 0.308043420315]],
                     [[0.163242310286, 0.308043420315]],
                     [[0.318325966597, 0.325517624617]],
                     [[0.447198301554, 0.321149080992]],
                     [[-0.674257874489, -0.151652157307]],
                     [[-0.170000001788, -0.075740583241]],
                     [[0.0, 0.0]],
                     [[0.170000001788, -0.075740583241]],
                     [[0.674257874489, -0.151652157307]],
                     [[-0.272456139326, -0.347239643335]],
                     [[0.0, -0.336318254471]],
                     [[0.272456139326, -0.347239643335]],
                     [[0.0, -0.737950384617]]], dtype=np.float32)
    #print tpl_pts.shape
    #print tpl_pts.dtype
    #print tpl_pts
    #print str(type(tpl_pts))
    tpl_rect = cv2.boundingRect(tpl_pts)
    # Mask points
    mask_idxs = np.where(pts_mask > 0.5)
    masked_pts = pts[mask_idxs]
    masked_tpl_pts = tpl_pts[mask_idxs]
    if masked_pts.shape[0] < 4 or masked_tpl_pts.shape[0] < 4:
        return (0, 0, 0, 0)


    # Homography matrix
    #print str(type(masked_tpl_pts))
    #print masked_tpl_pts.shape
    #print masked_tpl_pts.dtype
    #print masked_tpl_pts
    #print str(type(masked_pts))
    #print masked_pts.shape
    #print masked_pts.dtype
    #print masked_pts
    m2npa = shit2chocolate(masked_pts)
    H, _ = cv2.findHomography(masked_tpl_pts, m2npa, method=cv2.RANSAC)
    if H is None:
        return (0, 0, 0, 0)
    # Apply to rect
    x1, y1 = tpl_rect[0], tpl_rect[1]
    x2, y2 = tpl_rect[2] + x1, tpl_rect[3] + y1
    rect_pts = np.array([[x1, y1, 1.0],
                         [x1, y2, 1.0],
                         [x2, y1, 1.0],
                         [x2, y2, 1.0]], dtype=np.float32)
    rect_pts = H.dot(rect_pts.T).T
    rect_pts = rect_pts[:, 0:2] / rect_pts[:, 2].reshape(4, 1)
    # Convert points to rectangle
    min_xy = np.min(rect_pts, axis=0)
    max_xy = np.max(rect_pts, axis=0)
    rect = (min_xy[0], min_xy[1], max_xy[0] - min_xy[0], max_xy[1] - min_xy[1])

    # landmark's rect
    pts_rect = cv2.boundingRect(m2npa)
    # padding
    x, y, w, h = pts_rect
    pad_w = w * landmark_pad_rate
    pad_h = h * landmark_pad_rate
    pts_rect = (x - pad_w / 2.0, y - pad_h / 2.0, w + pad_w, h + pad_h)

    # union with landmark's rect
    rect = common.rect_or(rect, pts_rect)

    # intersect with img_rect
    rect = common.rect_and(rect, img_rect)
    return rect


def _bounding_region(pts, pts_mask):
    masked_pts = pts[np.where(pts_mask > 0.5)]
    if masked_pts.shape[0] == 0:
        return (0, 0, 0, 0)
    return cv2.boundingRect(shit2chocolate(masked_pts))


class HyperFace(object):
    def __init__(self, batchsize=32):
        # Define a model
        self.model = CnnFPGA(batchsize=batchsize)
        self.batchsize = batchsize

    def __call__(self, img):
        logger.info('Start analyzing image')

        # ========== Iterative Region Proposals (IRP) ==========
        img_rect = (0, 0, img.shape[1], img.shape[0])
        detections, landmarks, visibilities = None, None, None
        for stage_cnt in six.moves.xrange(3):
            if stage_cnt == 0:
                # Selective search, crop and normalize
                ssrects = common.selective_search_dlib(img,
                                                       max_img_size=(999, 999),
                                                       kvals=(50, 200, 2),
                                                       min_size=1500,
                                                       check=False,
                                                       debug_window=False)


            logger.info('Starting run ' + str(stage_cnt+1) + ' of 3 with ' + str(len(ssrects)) + ' initial proposed regions')
            # Forward
            detections, landmarks, visibilities, poses, genders = \
                _forward_with_rects(self.model, img, ssrects, self.batchsize)

            # Update ssrects using landmarks
            new_ssrects = list()
            for i in six.moves.xrange(len(ssrects)):
                if detections[i] > 0.25:  # TODO configure
                    new_ssrect = _proposal_region(landmarks[i],
                                                  visibilities[i], img_rect)
                    if new_ssrect[2] > 0 and new_ssrect[3] > 0:
                        new_ssrects.append(new_ssrect)
            ssrects = new_ssrects

        # [DEBUG] Draw IRP rectabgles
#         for rect in ssrects:
#             drawing._draw_rect(img, rect, (0, 1, 0))

        # Extract detected entries
        valid_idxs = [i for i, det in enumerate(detections) if det > 0.5]
        landmarks = np.asarray(landmarks)[valid_idxs]
        visibilities = np.asarray(visibilities)[valid_idxs]
        poses = np.asarray(poses)[valid_idxs]
        genders = np.asarray(genders)[valid_idxs]

        # ========== Landmark-based NMS ==========
        res_idx_sets = list()
        precise_rects = [_bounding_region(l, v) for l, v
                         in zip(landmarks, visibilities)]
        areas = [common.rect_area(rect) for rect in precise_rects]
        overlap_tls = 0.20  # TODO configure
        scorebase_idxs = np.argsort(areas).tolist()  # ascending order
        while len(scorebase_idxs) > 0:
            # Register new index set with the best rect index
            best_rect_idx = scorebase_idxs.pop()
            res_idx_sets.append([best_rect_idx])
            # Register overlapped indices
            best_rect = precise_rects[best_rect_idx]

#             # [DEBUG] Draw L-NMS rectabgles
#             drawing._draw_rect(img, best_rect, (0, 1, 0))

            removal_scorebase_idxs = list()
            for s_i, i in enumerate(scorebase_idxs):
                overlap = common.rect_overlap_rate(best_rect, precise_rects[i])
                if overlap > overlap_tls:
                    res_idx_sets[-1].append(i)
                    removal_scorebase_idxs.append(s_i)
            # Remove registered indices (reverse)
            for i in removal_scorebase_idxs[::-1]:
                scorebase_idxs.pop(i)
        # Extract middle value
        res_landmarks = list()
        res_visibilities = list()
        res_poses = list()
        res_genders = list()
        res_rects = list()
        for idx_set in res_idx_sets:
            res_landmarks.append(np.median(landmarks[idx_set], axis=0))
            res_visibilities.append(np.median(visibilities[idx_set], axis=0))
            res_poses.append(np.median(poses[idx_set], axis=0))
            res_genders.append(np.median(genders[idx_set], axis=0))
            res_rects.append(_proposal_region(res_landmarks[-1],
                                              res_visibilities[-1], img_rect))

        return (res_landmarks, res_visibilities, res_poses, res_genders,
                res_rects)
