import cv2 as cv
import numpy as np
import math


class LevelSet:
    def __init__(self, img):  # 初始化时输入源图像
        # 基本参数
        self.iterNum = 0  # 迭代次数

        self.global_item = 1.0  # 全局项系数
        self.length_bind = 0.001 * 255 * 255  # 长度约束系数
        self.penalty_item = 1.0  # 惩罚项系数
        self.time_step = 0.1  # 演化步长
        self.epsilon = 1.0  # 规则化参数
        self.depth = np.float32  # 深度

        # 过程数据
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

    def initialize(self, iter_number):
        """
        初始化函数
        :return: 无
        """
        # 取灰度
        self.gray_img = cv.cvtColor(src, cv.COLOR_RGB2GRAY)

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

        # 初始化前景水平集，内部置2，其余置-2
        for h in range(self.img_height):
            for w in range(self.img_width):
                self.m_Phi[h][w] = 2

    def evolution(self):
        """
        水平集演化函数
        :return: 最终结果
        """
        for i in range(self.iterNum):  # 迭代次数
            # 计算各种数组
            self.dirac()  # 狄拉克数组
            # print(self.m_Dirac)
            self.curvature()  # 曲率数组
            # print(self.m_Curvy)
            self.binary_fit()  # 计算前景、背景灰度均值
            # print(self.m_Heaviside)
            # print(self.FG_value, self.BK_value)

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
            # print(self.m_Phi)
            cv.imshow("final", self.m_Phi)

            # 显示每一次演化的结果
            # 转RGB
            # self.final_img = cv.cvtColor(self.gray_img, cv.COLOR_GRAY2RGB)
            # mask = self.final_img  # 找到掩膜
            #
            # print(mask)

            # print(mask)
            # cv.imshow("mask", mask)

            # # 膨胀腐蚀
            # kernel2 = cv.getStructuringElement(cv.MORPH_RECT, (1, 1))
            # mask = cv.morphologyEx(mask, cv.MORPH_ERODE, kernel2)  # 腐蚀
            # mask = cv.morphologyEx(mask, cv.MORPH_DILATE, kernel2)  # 膨胀

            # # 膨胀腐蚀
            # mask = cv.dilate(mask, (-1, -1))
            # mask = cv.erode(mask, (-1, -1))

            # 再次寻找轮廓
            # clone_image2, contours2, layout2 = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            # for c_i, contour in enumerate(contours2):
            #     cv.drawContours(self.final_img, contours2, c_i, (0, 255, 0), 1)
            # cv.imshow("final_contours", self.final_img)

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

        print(sum_fg, sum_bk, sum_h)

        # 前景灰度均值
        self.FG_value = sum_fg/(sum_h + 1e-10)
        # 背景灰度均值
        self.BK_value = sum_bk/(self.img_height * self.img_width - sum_h + 1e-10)


start = cv.getTickCount()  # 开始时间

# 输入输出
src = cv.imread("./resources/yuan.jpg")
# src = cv.imread("./resources/test.jpg")
cv.imshow("level_set_src", src)

# print(src)
print(src.shape)

# 初始化类
ls = LevelSet(src)

# 初始化各项值
ls.initialize(1)  # 迭代次数

# 水平集演化
ls.evolution()


end = cv.getTickCount()  # 结束时间
print("所用时间：%s 秒" % ((end - start)/cv.getTickFrequency()))

cv.waitKey(0)
cv.destroyAllWindows()
