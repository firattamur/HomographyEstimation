import cv2
import numpy as np


def warp_with_homography(src: np.ndarray, pts_src: np.ndarray, dst_shape: tuple, pts_dst: np.ndarray) -> np.ndarray:
    """
    Warp src image with calculate Homography matrix from src and dst points.

    src: source image
    src_pts: source points

    dst: destination image
    dst_pts: destination points match with source points.
    
    """
    dst_h, dst_w = dst_shape

    h, status = cv2.findHomography(pts_src, pts_dst)

    # warp source image to align planes in src and dst image
    warp = cv2.warpPerspective(src, h, (dst_h, dst_w))

    return warp


def collect_corners(filename: str) -> np.ndarray:
    """
    Collect four pixel coordinates for corners of any object in image.
    
    filename: path for source image
    """

    corners = ["LeftTop", "LeftBottom", "RightTop", "RightBottom"]
    pts_src = []
    window_name = "ClickTheCornersLeft->Top-Bottom|Right-Top->Bottom"
    completed = False
    
    def click(event, x, y, flags, params):

        nonlocal completed
        nonlocal corners

        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"{corners[len(pts_src)]}: {[x, y]}")
            pts_src.append([x, y])

        if len(pts_src) == 4:
            completed = True

    img = cv2.imread(filename)
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, click)

    while not completed:

        cv2.imshow(window_name, img)
        cv2.waitKey(1)

    cv2.destroyAllWindows()

    return np.array(pts_src)


def dst_points_with_aspect_ratio(aspect_ratio: str) -> np.ndarray:
    """
    Create image respect to aspect ratio:

    aspect_ratio: string representation of aspect ratio of image -> '300/400'
    """

    splits = aspect_ratio.split('/')
    width, height = int(splits[0]), int(splits[1])

    left_top = [0, 0]
    left_bottom = [0, height - 1]

    right_top = [width - 1, 0]
    right_bottom = [width - 1, height - 1]

    dst_pts = np.array([left_top, left_bottom, right_top, right_bottom])

    return dst_pts, (width, height)




 