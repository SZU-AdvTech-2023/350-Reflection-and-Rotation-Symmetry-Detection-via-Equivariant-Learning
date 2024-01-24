import cv2
import numpy as np

for i in range(3,13,1):
    path = "./images/refs_{:0>3}.jpg".format(i)
    myimg = "./images_result/refs_{:0>3}_reflection_axis.png".format(i)
    saveimg = "./drawhough_2/refs_{:0>3}_symmetry.jpg".format(i)

    # 读取图像并转为灰度图像
    img = cv2.imread(myimg)
    img2 = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 对图像进行二值化处理
    thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
    # 检测直线
    lines = cv2.HoughLinesP(thresh, rho=1, theta=np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
    # 绘制检测到的直线
    # for line in lines:
    #     x1, y1, x2, y2 = line[0]
    #     cv2.line(img2 , (x1,y1), (x2,y2), (0,255,0), 2)
    if len(lines) > 0:
        x1, y1, x2, y2 = lines[0][0]
        cv2.line(img2, (x1,y1), (x2,y2), (0,255,0), 2)
        print('refs_{:0>3}'.format(i),str(x1)+","+str(y1),str(x2)+","+str(y2))
   # 显示结果

