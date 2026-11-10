"""CPU scene signals at 2 fps (every 5th work frame): camera motion and table clutter.

  $VLA_PY -m cleantables.perception.scene.signals VIDEO...
cam_shift[i]: median displacement (px, 960-wide) of ORB matches (RANSAC homography inliers) between sample i-1 and i -> camera moves.
wood_frac[i]: fraction of 'bare wood' pixels (HSV range, measured on frames) inside the camera segment's table hull, i.e. 1 - clutter.
Out data/table_clean/_work/<video>/scene.npz: k (m,), cam_shift (m,), wood_frac (m,), seg (m,) camera segment id, segs [(k0,k1)].
"""
from __future__ import annotations

import sys

import cv2
import numpy as np

from cleantables.perception.scene.common import WORK, meta, read_frame

EVERY = 5
SHIFT_MOVE = 6.0      # px median match displacement between 0.5 s samples = camera moved


def wood_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    return (h >= 7) & (h <= 22) & (s >= 50) & (s <= 190) & (v >= 120)


def main(video):
    m = meta(video)
    ks = list(range(0, m["n"], EVERY))
    orb = cv2.ORB_create(1500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    prev = None
    shift = np.zeros(len(ks))
    woods = []
    for i, k in enumerate(ks):
        f = read_frame(video, k)
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        kp, de = orb.detectAndCompute(g, None)
        if prev is not None and de is not None and prev[1] is not None:
            mt = bf.match(prev[1], de)
            if len(mt) >= 12:
                a = np.float32([prev[0][x.queryIdx].pt for x in mt])
                b = np.float32([kp[x.trainIdx].pt for x in mt])
                H, inl = cv2.findHomography(a, b, cv2.RANSAC, 3.0)
                if H is not None:
                    inl = inl.ravel().astype(bool)
                    shift[i] = float(np.median(np.linalg.norm(b[inl] - a[inl], axis=1))) if inl.sum() >= 8 else 99
            else:
                shift[i] = 99
        prev = (kp, de)
        woods.append(cv2.resize(wood_mask(f).astype(np.uint8), (240, 135), interpolation=cv2.INTER_AREA) > 0)
    woods = np.array(woods)
    # camera segments: split where the camera moved; a moving phase (handheld adjustment) is excluded until it is still again
    moving = shift > SHIFT_MOVE
    seg = np.full(len(ks), -1)
    sid, cur = -1, False
    for i in range(len(ks)):
        if moving[i]:
            cur = False
            continue
        if not cur:
            sid += 1
            cur = True
        seg[i] = sid
    # merge tiny segments (< 5 s) into neighbours as -1 (transitional)
    segs = []
    for s in range(sid + 1):
        idx = np.where(seg == s)[0]
        if len(idx) * EVERY / m["fps"] < 5:
            seg[idx] = -1
        else:
            segs.append((int(ks[idx[0]]), int(ks[idx[-1]])))
    seg2 = np.full(len(ks), -1)
    for j, (a, b) in enumerate(segs):
        seg2[(np.array(ks) >= a) & (np.array(ks) <= b)] = j
    wood_frac = np.full(len(ks), np.nan)
    for j in range(len(segs)):
        idx = np.where(seg2 == j)[0]
        hull = woods[idx].mean(0) > 0.3        # pixels that are bare wood in >= 30 % of the segment = table area (clutter/hands come and go)
        hull = cv2.morphologyEx(hull.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)) > 0
        cnts, _ = cv2.findContours(hull.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        big = max(cnts, key=cv2.contourArea)
        hm = np.zeros(hull.shape, np.uint8)
        cv2.fillPoly(hm, [cv2.convexHull(big)], 1)
        hm = hm > 0
        for i in idx:
            wood_frac[i] = woods[i][hm].mean()
    np.savez(WORK / video / "scene.npz", k=np.array(ks), cam_shift=shift, wood_frac=wood_frac, seg=seg2, segs=np.array(segs))
    print(video, "camera segments (work frames):", [(a, b, f"{a / m['fps']:.0f}-{b / m['fps']:.0f}s") for a, b in segs], flush=True)


if __name__ == "__main__":
    for v in sys.argv[1:]:
        main(v)
