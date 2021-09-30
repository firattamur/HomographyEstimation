import cv2
import numpy as np


def warp_with_homography(src: np.ndarray, pts_src: np.ndarray, dst: np.ndarray, pts_dst: np.ndarray) -> np.ndarray:
    """
    Warp src image with calculate Homography matrix from src and dst points.

    src: source image
    src_pts: source points

    dst: destination image
    dst_pts: destination points match with source points.
    
    """
    dst_h, dst_w, dst_c = dst.shape

    h, status = cv2.findHomography(pts_src, pts_dst)

    # warp source image to align planes in src and dst image
    warp = cv2.warpPerspective(src, h, (dst_h, dst_w))

    return warp
 