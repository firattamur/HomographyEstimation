import cv2
import numpy as np
from utils.utils import warp_with_homography, collect_corners, dst_points_with_aspect_ratio


if __name__ == "__main__":
    
    file_src = "./assets/book.jpeg"

    # read source
    src = cv2.imread(file_src)

    # get corners of the object
    pts_src = collect_corners(filename=file_src)
    
    # get aspect ratio of the object
    aspect_ratio = input("Aspect Ratio of Object (eg. w/h): ")
    pts_dst, dst_shape = dst_points_with_aspect_ratio(aspect_ratio=aspect_ratio)

    # warp image by source and destination points
    warp = warp_with_homography(src=src, pts_src=pts_src, dst_shape=dst_shape, pts_dst=pts_dst)

    cv2.imshow("Source Image", src)
    cv2.imshow("Warped Image", warp)

    cv2.waitKey(0)
        

