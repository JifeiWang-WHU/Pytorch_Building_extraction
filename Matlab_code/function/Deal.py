import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from function.Algo import Point


def split(image):
    # 分量的提取
    ch1, ch2, ch3 = cv.split(image)
    return ch1, ch2, ch3


def k_means(image, k):
    # k_means实现，输入k值

    # k_means读入
    img_shape = image.reshape((-1, 3))
    # 转换为float类型
    data = np.float32(img_shape)
    # 定义标准
    criteria = (cv.TERM_CRITERIA_EPS, 10, 10.0)
    # 调用k_means函数
    ret, label, center = cv.kmeans(data, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

    # 转回uint8格式，恢复图像
    center = np.uint8(center)
    k_img_ori = center[label.flatten()]
    k_img = k_img_ori.reshape(image.shape)

    return k_img


def translate(image, src_image):
    # 去除阴影，所有像素遍历，黑白反转，与源图像srcIamge作加处理，实现将阴影变为白色
    # 第一个输入为反转图像，第二个为源图像

    # 遍历将图转为二值
    h, w, c = image.shape
    print(image.shape)
    for row in range(h):
        for col in range(w):
            if image[row, col, 0] == 0:
                image[row, col, 0] = 255
                image[row, col, 1] = 255
                image[row, col, 2] = 255
            else:
                image[row, col, 0] = 0
                image[row, col, 1] = 0
                image[row, col, 2] = 0

    # 与源图像进行加处理
    f_image = cv.add(image, src_image)
    print(image.shape)
    print(src_image.shape)

    return f_image


def canny(img):
    # canny边缘检测

    # 高斯模糊
    blurred = cv.GaussianBlur(img, (3, 3), 0)
    # 灰度图
    # gray = cv.cvtColor(blurred, cv.COLOR_RGB2GRAY)
    # 计算x、y方向梯度
    x_grad = cv.Sobel(blurred, cv.CV_16SC1, 1, 0)
    y_grad = cv.Sobel(blurred, cv.CV_16SC1, 0, 1)
    # canny边缘检测
    edge = cv.Canny(x_grad, y_grad, 50, 100)

    # 在原图基础上显示
    dst = cv.bitwise_and(img, img, mask=edge)

    return edge, dst


# 颜色区域提取
def color_area_red(self):
    # 提取红色区域(暂定框的颜色为红色)

    low_red = np.array([50, 60, 100])  # 为bgr
    high_red = np.array([85, 130, 180])
    mask = cv.inRange(self, low_red, high_red)
    red = cv.bitwise_and(self, self, mask=mask)
    return red


def color_area_blue(self):
    # 提取蓝色区域(暂定框的颜色为蓝色)

    low_red = np.array([90, 90, 60])  # 为bgr
    high_red = np.array([255, 220, 200])
    mask = cv.inRange(self, low_red, high_red)
    red = cv.bitwise_and(self, self, mask=mask)
    return red


def max_b(image):
    """
        取b值最大区域
    """
    h, w, c = image.shape
    for row in range(h):
        for col in range(w):
            # 判断是否B最大且大一部分
            if (image[row, col, 0] < image[row, col, 1] + 5) | (image[row, col, 0] < image[row, col, 2] + 5):
                image[row, col, 0] = image[row, col, 1] = image[row, col, 2] = 0
    return image


def show_value(image, n):
    """
        突出显示某值
        R、G、B的n值分别为2、1、0；H、S、V的n值分别为0、1、2
    """

    h, w = image.shape[:2]
    for i in range(h):
        for j in range(w):
            image[i, j, n] = 255
    return image


def image_hist(image, masks):
    """
        绘制三通道直方图
        其中RGB的直方图分别为红绿蓝色；HSV的直返图分别为蓝绿红色
        :param image: 三通道图像
        :return: 直接绘制
    """

    # 绘制的颜色
    colors = ('blue', 'green', 'red')
    for i, color in enumerate(colors):
        hist = cv.calcHist([image], [i], masks, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    plt.show()


def point_out(point):
    """
    输出point类点的各值
    :param point: point类
    :return: 无
    """
    print("x:%s, y:%s, value:%s, group:%s" % (point.x, point.y, point.value, point.group))


def to_binary(img, flag):
    """
    将聚类结果变为二值化（当k值为2时）
    :param img: kmeans后的结果
    :param flag: 需要反转标志位，为0时不需要，为1时需要
    :return: 黑白二值化
    """
    width, height = img.shape  # 获取长宽

    # 判断标志位
    if flag == 0:
        swallow = 255
        deep = 0
    else:
        swallow = 0
        deep = 255

    for w in range(width):  # 遍历每一个点
        for h in range(height):
            if img[w][h] == 2:
                img[w][h] = deep
            else:
                img[w][h] = swallow
    return img


def contours_to_roi(img, contour):
    """
    获取一个轮廓曲线的原图ROI区域
    :param img: 原图
    :param contour: 轮廓
    :return: ROI区域和已经消去的轮廓和x、y偏移值
    """
    # 求得上下左右最大边界，其中位置为：
    #     ...........top...........
    #     .                       .
    #   left                    right
    #     .                       .
    #     ...........down..........
    # 初始化为第一个点的值
    left = right = contour[0][0][0]
    top = down = contour[0][0][1]

    # 遍历获得四个角点，其中contour[i][0][0]表示横坐标，contour[i][0][1]表示纵坐标
    for i in range(len(contour)):
        # 更新角点
        if contour[i][0][0] < left:
            left = contour[i][0][0]
        elif contour[i][0][0] > right:
            right = contour[i][0][0]
        if contour[i][0][1] > down:
            down = contour[i][0][1]
        elif contour[i][0][1] < top:
            top = contour[i][0][1]
    # 消去轮廓
    for i in range(len(contour)):
        # 挨个减去最左最上值
        contour[i][0][0] = contour[i][0][0] - left
        contour[i][0][1] = contour[i][0][1] - top

    # ROI区域稍微放大，而且要避免越界
    # if left - 5 < 0:  # 先来5个像素的
    #     left = 0
    # else:
    #     left = left - 5
    #
    # if top - 5 < 0:  # 先来5个像素的
    #     top = 0
    # else:
    #     top = top - 5
    #
    # if right + 5 > img.shape[1]:  # 先来5个像素的
    #     right = img.shape[1]
    # else:
    #     right = right + 5
    #
    # if down + 5 > img.shape[0]:  # 先来5个像素的
    #     down = img.shape[0]
    # else:
    #     down = down + 5

    # bug记录4：直接给img取ROI时先是纵坐标再是横坐标
    img_roi = img[top:down, left:right].copy()
    # 以四个角点返回roi区域
    return img_roi, contour, left, top


def contours_to_area(roi, contour):
    """
    将获取到的ROI区域进行填充，并且消除细小轮廓，返回一个包括该轮廓的ROI
    :param roi:轮廓的ROI区域
    :param contour:提取到的轮廓
    :return:进行填充
    """
    # 对于轮廓进行消除或者填充
    area = cv.contourArea(contour)  # 计算区域面积
    height, width, channels = roi.shape  # 获得高和宽
    shadow = np.zeros([height, width], np.uint8)

    # 处理过小面积或者进行填充
    if area < 30:  # 先试一下
        c_min = []  # 消除参数
        # thickness为-1时表示填充，填充为黑色表示消除

        c_min.append(contour)
        cv.drawContours(shadow, c_min, -1, (0, 0, 0), thickness=-1)
    else:
        c_max = []
        # 轮廓填充为白色
        c_max.append(contour)
        cv.drawContours(shadow, c_max, -1, (255, 255, 255), thickness=-1)

    return shadow
