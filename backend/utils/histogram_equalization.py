# 使用CLAHE算法进行光照预处理
import numpy as np
import cv2 as cv
from PIL import Image


# 彩色图像进行自适应直方图均衡化
def hisEqulColor(img):
    img = np.array(img)
    # 将RGB图像转换到YCrCb空间中, Y为颜色的亮度成分、而CB和CR则为蓝色和红色的浓度偏移量成分
    ycrcb = cv.cvtColor(img, cv.COLOR_RGB2YCR_CB)
    # 将YCrCb图像通道分离
    channels = cv.split(ycrcb)
    # 以下代码详细注释见官网：
    # https://docs.opencv.org/4.1.0/d5/daf/tutorial_py_histogram_equalization.html
    # 图像被分成称为"titles"的小块,默认大小为8x8,对每一个小块进行直方图均衡化,clipLimit参数和对比度限制有关
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # 对第1个通道即亮度通道进行全局直方图均衡化,作为输出
    clahe.apply(channels[0], channels[0])
    cv.merge(channels, ycrcb)
    img_rgb = Image.fromarray(cv.cvtColor(ycrcb, cv.COLOR_YCR_CB2RGB))
    return img_rgb


if __name__ == '__main__':
    # 测试代码
    # 返回的是Image对象,读入的顺序为RGB
    img = Image.open('braised pork.jpg')
    res = hisEqulColor(img)  # 自适应直方图均衡化后的图res1
    res.save('./after-process.jpg')
