import random
import numpy as np
import math
import cv2 as cv
from function import Deal

# 类：点
class Point:
    # x和y是二维数组的坐标，value是其单通道值，group为该点的聚类类型
    # 其中x对应width值，y对应height值
    __slots__ = ["x", "y", "value", "group"]

    # 初始化归零
    def __init__(self, x=0, y=0, value=0, group=0):
        self.x, self.y, self.value, self.group = x, y, value, group


# 最大距离：无限大
DISTANCE_MAX = 1e100


def p2p_distance(point_a, point_b):
    """
    求两个Point点之间的距离
    :param point_a: Point类点a
    :param point_b: Point类点b
    :return: 距离平方
    """
    # bug记录1：两个value都是无符号数，无符号数直接相减肯定会出现问题，所以先判断大小
    if point_a.value > point_b.value:
        result = point_a.value - point_b.value
    else:
        result = point_b.value - point_a.value
    return result ** 2


def nearest_center(point, cluster_centers):
    """
    求一个点的最近中心点
    :param point: 目标点，Point类
    :param cluster_centers: 各个中心点，为一个list，输入之前的已有初始点，list内元素为Point类
    :return: 最小索引和最小距离
    """

    # 设定最大最小值
    min_index = point.group
    min_dist = DISTANCE_MAX

    # 递归比较，找到最小的那个中心点
    for i, p in enumerate(cluster_centers):  # i为在cluster_centers里的索引，p为该Point点
        dis = p2p_distance(p, point)
        if min_dist > dis:
            min_dist = dis
            min_index = i

    return min_index, min_dist


def k_means_plus(img, cluster_centers):
    """
    k_means++初始化中心点函数
    :param img:数据集，二维数组，或者是Mat形式，单通道
    :param cluster_centers: 聚类初始点，是一个list，list内是Point类
    :return:返回List存放的k个初始点
    """
    # 获取图像高宽
    height, width = img.shape
    # 随机选取一个为中心点
    x0 = random.randint(0, width-1)
    y0 = random.randint(0, height-1)
    # 将随机值赋给初始第一个Point点的x和y值
    cluster_centers[0].x, cluster_centers[0].y = x0, y0
    cluster_centers[0].value = img[y0][x0]  # 为初始点一号赋value值
    # print(cluster_centers[0].x, cluster_centers[0].y, cluster_centers[0].value)

    # 初始化存数每个数据点到最近中心点的距离的Mat数组
    distance = np.zeros([height, width], np.uint16)  # 定义一个同等大小的数组，初始归0

    # 递归求初始化
    for i in range(1, len(cluster_centers)):  # k-1次递归，k为聚类数
        # 计算每个点到中心质点的距离并储存
        # bug记录2：好像是有符号数达到了最大尺寸的一半会导致溢出，所以改用numpy的ulonglong类型
        cc_sum = np.ulonglong(0)  # 保存每个中心点的总和

        for h in range(height):  # 遍历每一个点
            for w in range(width):
                temporarily_point = Point(w, h, img[h][w])  # 取到目前点的坐标和值
                distance[h][w] = nearest_center(temporarily_point, cluster_centers[:i])[1]  # 计算该点离最近中心点的距离并储存
                cc_sum += distance[h][w]  # 存储总距离

        # print(distance)
        # print(cc_sum)

        # 化用公式(distance[i]/cc_sum)，并用一个0到1之间的随机数遍历进行寻找
        random_cc_sum = cc_sum * random.random()  # 化用第一步，直接给总数乘以0到1的随机数
        # print("random_cc_sum:%s" % random_cc_sum)

        break_flag = False  # 跳出多层循环的标志
        for h2 in range(height):  # 再次遍历每一个点
            for w2 in range(width):
                random_cc_sum -= distance[h2][w2]  # 如果小于0说明在这个区间内
                if random_cc_sum > 0:
                    continue
                else:
                    cluster_centers[i] = Point(w2, h2, img[h2][w2])  # 获得其点存入质心数组
                    break_flag = True
                    break
            if break_flag:
                break

    return cluster_centers


def k_means(src, k, cluster_centers, iteration_number=0, type_flag=0):
    """
    k_means单通道聚类
    :param src: 输入图像
    :param k: 聚类数
    :param cluster_centers:
    :param iteration_number: 聚类次数，默认为0，表示不使用聚类次数
    :param type_flag: 输入图像的类型，V值为0， S值为1
    :return: 返回聚类后的图像
    """
    changed = True  # 以聚类点是否变换判断是否收敛
    # 获取图像宽高
    height, width = src.shape
    # 初始化存数每个数据点聚类类型的Mat数组
    groups = np.zeros([height, width], np.uint8)  # 定义一个同等大小的数组，初始归0

    # 先将初始聚类点存入聚类类型数组，并且加1存储（与初始值区别）
    for i, p in enumerate(cluster_centers):
        groups[p.y][p.x] = i + 1

    # 没有迭代次数时设置标志
    if iteration_number == 0:
        iteration_number_flag = 1
    else:
        iteration_number_flag = 0

    # 要么迭代次数到达，要么不再变化
    while changed & (bool(iteration_number_flag) | ((not iteration_number_flag) & bool(iteration_number))):
        changed = False
        iteration_number -= 1

        # 对于每一个点计算离其最近的聚类
        for h in range(height):
            for w in range(width):
                temporarily_point = Point(w, h, src[h][w], groups[h][w])  # 取到目前点的坐标、值和聚类类型

                min_index2 = groups[h][w]  # 取此点原始距离
                min_dist2 = DISTANCE_MAX  # 准备存储最小距离

                # 递归比较，找到最小的那个中心点索引
                for i, p in enumerate(cluster_centers):  # i为在cluster_centers里的索引，p为该Point点
                    dis2 = p2p_distance(p, temporarily_point)
                    if min_dist2 > dis2:
                        min_dist2 = dis2
                        min_index2 = i+1

                # 判断索引是否有变化
                if groups[h][w] != min_index2:
                    changed = True
                    groups[h][w] = min_index2

        # 更新每个聚类的中心点
        for k_i in range(k):
            v_sum = 0  # 存储值总和
            v_count = 0  # 聚类总数

            # 同一聚类值相加
            for h2 in range(height):
                for w2 in range(width):
                    if k_i + 1 == groups[h2][w2]:
                        v_sum += src[h2][w2]
                        v_count += 1

            # 更新中心点
            center_value = v_sum/v_count
            cluster_centers[k_i].value = center_value

    # 保证浅色为白色，深色为黑色
    if cluster_centers[0].value > cluster_centers[1].value:  # 用[0]存储深色类
        binary_inv_flag = 0  # 反转标志正常
    else:
        binary_inv_flag = 1  # 反转标志异常，需要反转

    # 根据输入图像类型判断反转
    if type_flag == 1:  # 若为S图，得再反转，保证[0]为深色类
        binary_inv_flag = type_flag - binary_inv_flag

    return groups, binary_inv_flag


class LevelSet:
    def __init__(self, img):  # 初始化时输入源图像
        # 基本参数
        self.iterNum = 1  # 迭代次数

        self.global_item = 1.0  # 全局项系数
        self.length_bind = 0.001 * 255 * 255  # 长度约束系数
        self.penalty_item = 1.0  # 惩罚项系数
        self.time_step = 0.001  # 演化步长
        self.epsilon = 1.0  # 规则化参数
        self.depth = np.float32  # 深度

        # 过程数据
        self.src = img  # 源图像
        self.gray_img = img  # 获取源灰度图像
        self.final_img = img  # 最终结果

        self.img_height = img.shape[0]  # 获取高
        self.img_width = img.shape[1]  # 获取宽

        self.FG_value = 0.0  # 前景值
        self.BK_value = 0.0  # 背景值

        self.m_Phi = img  # 水平集

        self.m_Dirac = img  # 狄拉克数组
        self.m_Heaviside = img  # 海氏函数数组
        self.m_Curvy = img  # 曲率数组
        self.m_Penalize = img  # 惩罚项数组

        self.penalty_kernel = np.zeros([3, 3], self.depth)  # 惩罚性卷积核

    def initialize(self, iter_number, roi, contour):
        """
        初始化函数，包括对曲线进行填充
        :param roi: 感兴趣区域
        :param contour: 轮廓
        :return: 无
        """
        # 取灰度
        self.gray_img = cv.cvtColor(self.src, cv.COLOR_RGB2GRAY)

        # 初始化水平集属性
        self.iterNum = iter_number  # 迭代次数

        # 初始化水平集数组
        self.final_img = self.gray_img  # 最终结果

        self.m_Phi = np.zeros([self.gray_img.shape[0], self.gray_img.shape[1]], np.float32)  # 水平集数组

        self.m_Dirac = np.zeros([self.gray_img.shape[0], self.gray_img.shape[1]], np.float32)  # 狄拉克数组
        self.m_Heaviside = np.zeros([self.gray_img.shape[0], self.gray_img.shape[1]], np.float32)  # 海氏函数数组
        self.m_Curvy = np.zeros([self.gray_img.shape[0], self.gray_img.shape[1]], np.float32)  # 曲率数组
        self.m_Penalize = np.zeros([self.gray_img.shape[0], self.gray_img.shape[1]], np.float32)  # 惩罚项数组

        # 初始化惩罚性卷积核
        self.penalty_kernel[0][0] = self.penalty_kernel[0][2] = 0.5
        self.penalty_kernel[2][0] = self.penalty_kernel[2][2] = 0.5
        self.penalty_kernel[0][1] = self.penalty_kernel[1][0] = 1
        self.penalty_kernel[1][2] = self.penalty_kernel[2][1] = 1
        self.penalty_kernel[1][1] = -6

        # 将ROI区域中的曲线区域进行填充
        roi = Deal.contours_to_area(roi, contour)

        # 二值化
        ret, roi_2value = cv.threshold(roi, 172, 255, cv.THRESH_BINARY)

        # 初始化前景水平集，内部置2，其余置-2
        for h in range(self.img_height):
            for w in range(self.img_width):
                if roi_2value[h][w] > 170:  # 若为白色置2
                    self.m_Phi[h][w] = 2
                else:
                    self.m_Phi[h][w] = -2

    def evolution(self):
        """
        水平集演化函数
        :return: 最终结果
        """
        for i in range(self.iterNum):  # 迭代次数
            # 计算各种数组
            self.dirac()  # 狄拉克数组
            self.curvature()  # 曲率数组
            self.binary_fit()  # 计算前景、背景灰度均值

            # 惩罚项卷积，计算惩罚项数组
            cv.filter2D(self.m_Phi, cv.CV_32FC1, self.penalty_kernel, self.m_Penalize, (1, 1))

            # 行列遍历计算
            for h in range(self.img_height):
                for w in range(self.img_width):
                    # 获取各个值
                    f_curvy = self.m_Curvy[h][w]
                    f_dirac = self.m_Dirac[h][w]
                    f_penalize = self.m_Penalize[h][w]
                    f_img_value = self.gray_img[h][w]

                    # 各项计算
                    length_term = self.length_bind * f_dirac * f_curvy  # 长度约束
                    penalize_term = self.penalty_item * (f_penalize - f_curvy)  # 惩罚项
                    temp1 = (-((f_img_value - self.FG_value) ** 2))
                    temp2 = ((f_img_value - self.BK_value) ** 2)
                    area_term = f_dirac * self.global_item * (temp1 + temp2)  # 全局项

                    # 更新水平集数组
                    self.m_Phi[h][w] += self.time_step * (length_term + penalize_term + area_term)

            return self.m_Phi

    def dirac(self):  # 狄拉克函数
        """
        获得狄拉克数组
        :return:
        """
        k1 = self.epsilon / math.pi
        k2 = self.epsilon ** 2
        for h in range(self.img_height):  # 遍历求出狄拉克数组
            for w in range(self.img_width):
                self.m_Dirac[h][w] = k1 / (k2 + self.m_Phi[h][w] ** 2)

    def heaviside(self):  # 海氏函数
        """
        获得海氏函数
        :return:
        """
        k3 = 2 / math.pi
        for h in range(self.img_height):  # 遍历求出海氏数组
            for w in range(self.img_width):
                self.m_Heaviside[h][w] = 0.5 * (1 + k3 * math.atan(self.m_Phi[h][w] / self.epsilon))

    def curvature(self):  # 曲率函数
        """
        计算曲率
        :return:
        """
        # 先计算一波梯度
        dx = cv.Sobel(self.m_Phi, cv.CV_32FC1, 1, 0, 1)
        dy = cv.Sobel(self.m_Phi, cv.CV_32FC1, 0, 1, 1)

        # 更新dx与dy
        for h in range(self.img_height):  # 遍历
            for w in range(self.img_width):
                val = math.sqrt(dx[h][w] ** 2 + dy[h][w] ** 2 + 1e-10)
                dx[h][w] = dx[h][w] / val
                dy[h][w] = dy[h][w] / val

        # 再次计算求得曲率
        ddy = cv.Sobel(dx, cv.CV_32FC1, 0, 1, 1)
        ddx = cv.Sobel(dy, cv.CV_32FC1, 1, 0, 1)
        # 加为曲率
        self.m_Curvy = ddx + ddy

    def binary_fit(self):  # 计算前景与背景值
        """
        计算前景与背景值
        :return:
        """
        # 先获得海氏数组
        self.heaviside()

        # 初始化值
        sum_fg = 0
        sum_bk = 0
        sum_h = 0

        # bug记录3：对照公式时将从1开始写成了从0开始，导致算不出较大的梯度差值
        #           同时opencv在vs和pycharm中对于同一幅图像素的提取值不同，有一定差异
        for h in range(1, self.img_height):  # 遍历
            for w in range(1, self.img_width):
                f_img_value = self.gray_img[h][w]
                f_heaviside = self.m_Heaviside[h][w]
                ff_heaviside = 1 - f_heaviside

                # 累加值
                sum_fg += f_img_value * f_heaviside
                sum_bk += f_img_value * ff_heaviside
                sum_h += f_heaviside

        # 前景灰度均值
        self.FG_value = sum_fg/(sum_h + 1e-10)
        # 背景灰度均值
        self.BK_value = sum_bk/(self.img_height * self.img_width - sum_h + 1e-10)
