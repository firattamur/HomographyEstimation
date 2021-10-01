import cv2
import numpy as np
from utils.utils import warp_with_homography, collect_corners, dst_points_with_aspect_ratio


if __name__ == "__main__":
    
    # import pdb
    # pdb.set_trace()
	
    # file path of source image
    filename_src = './assets/ad-image.jpeg'

    # src image
    src = cv2.imread(filename_src)

    cv2.imshow("Source Image", src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # shape of src
    src_h, src_w, src_c = src.shape

    # source points
    pts_src = np.array([
        [0, 0],
        [0, src_h - 1],
        [src_w - 1, src_h - 1],
        [src_w - 1, 0],
    ])

    # file path of target image
    filename_dst = './assets/virtual-bilboard.jpeg'

    # dst image
    dst = cv2.imread(filename_dst)
    dst_shape = (dst.shape[1], dst.shape[0])

    # get corners of the bilboard
    pts_dst = collect_corners(filename=filename_dst)

    # warp image
    warp = warp_with_homography(src=src, dst_shape=dst_shape, pts_src=pts_src, pts_dst=pts_dst)

    # black out polygonal area in destination image.
    blackout = cv2.fillConvexPoly(dst, pts_dst.astype(int), 0, 16);
 
    warp = blackout + warp

    cv2.imshow("Warped Image", warp)
    cv2.waitKey(0)
