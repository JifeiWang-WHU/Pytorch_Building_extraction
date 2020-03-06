import cv2 as cv


def img_in(str_path):
    # 图片输入
    image = cv.imread(str_path)
    return image


def img_out(win_name, image):
    # 图片输出
    cv.imshow(win_name, image)