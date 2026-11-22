"""Descriptor agreement on a *given pair* of images. No map search or pose output."""
from time import perf_counter

import cv2
import numpy as np


def gray(image):
    if image is None or image.size == 0:
        return None
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def distinctive_neighbors(matcher, source, target):
    accepted = {}
    for neighbors in matcher.knnMatch(source, target, k=2):
        if len(neighbors) == 2:
            first, second = neighbors
            if first.distance < 0.8 * second.distance:
                accepted[first.queryIdx] = first.trainIdx
    return accepted


def score_pair(query, reference, method, clahe_clip=0):
    """Count reciprocal distinctive descriptor matches; points use input pixel units."""
    start = perf_counter()
    detectors = {"ORB": cv2.ORB_create, "AKAZE": cv2.AKAZE_create}
    detector = detectors[method]()
    left, right = gray(query), gray(reference)
    points = []
    if left is not None and right is not None and min(*left.shape, *right.shape) >= 32:
        if clahe_clip:
            left = cv2.createCLAHE(float(clahe_clip), (8, 8)).apply(left)
        kp1, desc1 = detector.detectAndCompute(left, None)
        kp2, desc2 = detector.detectAndCompute(right, None)
        if desc1 is not None and desc2 is not None and min(len(desc1), len(desc2)) >= 2:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
            forward = distinctive_neighbors(matcher, desc1, desc2)
            reverse = distinctive_neighbors(matcher, desc2, desc1)
            points = [[*kp1[i].pt, *kp2[j].pt] for i, j in forward.items() if reverse.get(j) == i]
    return {"score": len(points), "inliers": len(points), "points": points,
            "algorithm_ms": (perf_counter() - start) * 1000}


def average_precision(labels, scores):
    """Non-interpolated AP with tied scores evaluated at one threshold."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    if len(labels) != len(scores) or not len(labels) or not labels.sum():
        raise ValueError("AP requires aligned scores and at least one positive")
    order = np.argsort(-scores, kind="stable")
    sorted_scores, sorted_labels = scores[order], labels[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(scores) - 1]
    tp = np.cumsum(sorted_labels)[ends]
    return float(np.sum(np.diff(np.r_[0, tp]) * tp / (ends + 1)) / labels.sum())
