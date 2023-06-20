# 数据增强脚本
# 在ubuntu系统上需执行apt-get update & apt-get install libglib2.0-dev
# 确保图像目录下面没有.ipynb_checkpoints这种无关文件，要删干净
import cv2
import numpy as np
import os.path
import copy
import os
 
# 昏暗
def darker(image,percetage=0.9):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    #get darker
    for xi in range(0,w):
        for xj in range(0,h):
            image_copy[xj,xi,0] = int(image[xj,xi,0]*percetage)
            image_copy[xj,xi,1] = int(image[xj,xi,1]*percetage)
            image_copy[xj,xi,2] = int(image[xj,xi,2]*percetage)
    return image_copy
 
# 翻转
def flip(image):
    flipped_image = np.fliplr(image)
    return flipped_image
 
 
# 样本数量翻两倍
def double_size(file_dir):
    for img_name in os.listdir(file_dir):
        img_path = file_dir + img_name
        img = cv2.imread(img_path)
        # 镜像
        flipped_img = flip(img)
        cv2.imwrite(file_dir + img_name[0:-4] + '_fli.jpg', flipped_img) # 保存图片
 
 
# 样本数量翻三倍
def triple_size(file_dir):
    for img_name in os.listdir(file_dir):
        img_path = file_dir + img_name
        img = cv2.imread(img_path)
        # 镜像
        flipped_img = flip(img)
        cv2.imwrite(file_dir +img_name[0:-4] + '_fli.jpg', flipped_img) # 保存图片
        # 变暗
        img_darker = darker(img)
        cv2.imwrite(file_dir+ img_name[0:-4] + '_darker.jpg', img_darker)
 
 
# 图片文件夹根目录
images_path = '/hy-tmp/Fruits_Veg_Dishes/images'
for sub_dir in sorted(os.listdir(images_path)):
    sample_size = len(os.listdir(images_path+'/'+sub_dir))
    
    if sample_size <= 400:
        triple_size(images_path+'/'+sub_dir+'/')
        print(images_path+'/'+sub_dir+'/')
    
    if sample_size > 400 and sample_size <= 600:
        double_size(images_path+'/'+sub_dir+'/')
        print(images_path+'/'+sub_dir+'/')
 
print('done.')