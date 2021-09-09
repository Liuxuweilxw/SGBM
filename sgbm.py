import cv2
import numpy as np
from matplotlib import pyplot as plt
import camera_configs

cv2.namedWindow("left")
cv2.namedWindow("right")
cv2.namedWindow("depth")

cv2.createTrackbar("num", "depth", 1, 11, lambda x: None)
cv2.createTrackbar("blockSize", "depth", 3, 255, lambda x: None)

# # 添加点击事件，打印当前点的距离
# def callbackFunc(e, x, y, f, p):
#     if e == cv2.EVENT_LBUTTONDOWN:
#         print(threeD[y][x])
#
#
# cv2.setMouseCallback("depth", callbackFunc, None)

while True:
    frame1 = cv2.imread("resources/1.png")
    frame2 = cv2.imread("resources/2.png")

    # 根据更正map对图片进行重构
    img1_rectified = cv2.remap(frame1, camera_configs.left_map1, camera_configs.left_map2, cv2.INTER_LINEAR)
    img2_rectified = cv2.remap(frame2, camera_configs.right_map1, camera_configs.right_map2, cv2.INTER_LINEAR)

    # 将图片置为灰度图，为StereoBM作准备
    imgL = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(img2_rectified, cv2.COLOR_BGR2GRAY)

    # 两个trackbar用来调节不同的参数查看效果
    num = cv2.getTrackbarPos("num", "depth")
    blockSize = cv2.getTrackbarPos("blockSize", "depth")

    if blockSize % 2 == 0:
        blockSize += 1
    if blockSize < 3:
        blockSize = 3

    # stereo = cv2.StereoBM_create(numDisparities=16 * num, blockSize=blockSize)
    stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=16 * num, blockSize=blockSize, P1=72,
                                   P2=288, disp12MaxDiff=1, speckleWindowSize=100, speckleRange=32, uniquenessRatio=10,
                                   preFilterCap=63)
    disparity = stereo.compute(imgL, imgR)

    disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # 将图片扩展至3d空间中，其z方向的值则为当前的距离
    # threeD = cv2.reprojectImageTo3D(disparity.astype(np.float32) / 16., camera_configs.Q)

    cv2.imshow("left", img1_rectified)
    cv2.imshow("right", img2_rectified)
    cv2.imshow("depth", disp)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break
