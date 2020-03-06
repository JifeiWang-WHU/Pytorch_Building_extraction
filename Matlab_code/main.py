"""
    毕业设计
    基于主动轮廓的卫星图像建筑物提取
"""
import cv2 as cv

# 导入其他文件函数库
from function import IO
from function import Deal

"""----------------------输入输出原图----------------------"""
# src = IO.img_in("./resources/map2.jpg")  # 测试图
src = IO.img_in("../NotPush/map3.jpg")
IO.img_out("original", src)

"""---------------------消除绿植和阴影---------------------"""
# 转为hsv
hsv = cv.cvtColor(src, cv.COLOR_RGB2HSV)

# 突出显示H值
hsv = Deal.show_value(hsv, 0)  # hsv图像0取H值
# IO.img_out("hsv", hsv)

# k_means实现
k_img = Deal.k_means(hsv, 2)
# IO.img_out("k_means", k_img)

# 二值化
k_img_gray = cv.cvtColor(k_img, cv.COLOR_RGB2GRAY)
# IO.img_out("gray", k_img_gray)
print(k_img_gray)
ret0, shadow_2value = cv.threshold(k_img_gray, 140, 255, cv.THRESH_BINARY_INV)  # 其中阈值在map2中取112到151之间的值，如140

# 转回三通道
shadow = cv.cvtColor(shadow_2value, cv.COLOR_GRAY2RGB)
# IO.img_out("shadow", shadow)

# 与源图像进行加处理
NoShadowImg = cv.bitwise_and(shadow, src)
IO.img_out("NoShadow", NoShadowImg)

"""-----------------------求初始曲线-----------------------"""
# 输出rgb
# r, g, b = Deal.split(NoShadowImg)
# IO.img_out("r", r)
# IO.img_out("g", g)
# IO.img_out("b", b)

# 转为hsv
NS_hsv = cv.cvtColor(NoShadowImg, cv.COLOR_RGB2HSV)
# IO.img_out("NS_hsv", NS_hsv)
# 输出hsv
v, s, h = Deal.split(NS_hsv)
# IO.img_out("h", h)
IO.img_out("s", s)
# IO.img_out("v", v)

# s值二值化
print(s)
ret1, s_2value = cv.threshold(s, 80, 255, cv.THRESH_BINARY_INV)  # 其中阈值在map2中取112到151之间的值，如140
cv.imshow("s_2value", s_2value)

# 突出显示HSV值
# IO.img_out("H2", Deal.show_value(NS_hsv, 0))
# IO.img_out("S2", Deal.show_value(NS_hsv, 1))
# IO.img_out("V2", Deal.show_value(NS_hsv, 2))

# 显示hsv直方图
# Deal.image_hist(NS_hsv, shadow_2value)
# 显示RGB直方图
# Deal.image_hist(src, shadow_2value)


# # 流出RGB中偏蓝色的区域
# test = Deal.color_area_blue(NoShadow_img)
# IO.img_out("test", test)
#
# # 保留B值最大的区域
# test_b = Deal.max_b(test)
# IO.img_out("test_b", test_b)

# 输出一个图像
# cv.imwrite("../NotPush/out1.jpg", NoShadow_img)

# # 转灰度图
# test_gray = cv.cvtColor(test_b, cv.COLOR_RGB2GRAY)
#
# # 将图二值化
# ret1, _2value = cv.threshold(test_gray, 110, 255, cv.THRESH_BINARY)
# IO.img_out("2value", _2value)
#
# # 对二值化的图进行开运算
# kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
# opened = cv.morphologyEx(_2value, cv.MORPH_OPEN, kernel)
# IO.img_out("Open", opened)
#
# # 开运算后闭运算，优化线条
# closed = cv.morphologyEx(opened, cv.MORPH_CLOSE, kernel)
# # 显示腐蚀后的图像
# IO.img_out("Close", closed)
#
# # 轮廓边缘检测
# canny_out, img3 = Deal.canny(closed)
# IO.img_out("canny", canny_out)
#
# # 轮廓线加到原图上
# canny_out_color = cv.cvtColor(canny_out, cv.COLOR_GRAY2RGB)
# extraction1 = cv.add(canny_out_color, img)
# cv.imshow("extraction1", extraction1)

# 等待
cv.waitKey(0)
cv.destroyAllWindows()