import cv2
import numpy as np
from utils.utils import warp_with_homography


if __name__ == "__main__":
    
    # src image
    img_src = cv2.imread('./assets/book2.jpeg')

    # four corners of src 
    pts_src = np.array([[141, 131], [480, 159], [493, 630], [64, 601]])

    # dest image
    img_dst = cv2.imread('./assets/book1.jpeg')
    dst_shape = (img_dst.shape[0], img_dst.shape[1])

    # four corners of dst
    pts_dst = np.array([[318, 256], [534, 372], [316, 670], [73, 473]])

    img_warp = warp_with_homography(img_src, pts_src, dst_shape, pts_dst)

    cv2.imshow("Source Image", img_src)
    cv2.imshow("Destination Image", img_dst)
    cv2.imshow("Warped Source Image", img_warp)

    cv2.waitKey(0)
        


