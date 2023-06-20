# 脚本作用：划分训练集、验证集、测试集
import os
import random
import shutil
 
# 源数据集的根目录
original_img_Dir = '/hy-tmp/Fruit-262/'
# 先在/hy-tmp/目录新建以下三个目录
target_train_Dir = '/hy-tmp/train'
target_val_Dir = '/hy-tmp/val'
target_test_Dir = '/hy-tmp/test'
train_radtio = 0.8
val_radtio = 0.1
test_radtio = 0.1
 
classname = os.listdir(original_img_Dir)
for class_folder in classname:
    #对其中的一个类别进行划分
    epath = os.path.join(original_img_Dir,class_folder) #路径
    e_nums = len(os.listdir(epath))       #每一类的图像数量
    train_nums = int(e_nums*train_radtio) # 训练集的数量
    val_nums = int(e_nums*val_radtio)     # 验证集的数量
    test_nums = int(e_nums*test_radtio)   # 测试集的数量
 
    train_save = os.path.join(target_train_Dir,class_folder)
    val_save = os.path.join(target_val_Dir,class_folder)
    test_save = os.path.join(target_test_Dir,class_folder)
    
    # 若没有建立该文件夹
    if not (os.path.exists(os.path.join(train_save))):
        os.mkdir(os.path.join(train_save))
    if not (os.path.exists(os.path.join(val_save))):
        os.mkdir(os.path.join(val_save))
    if not (os.path.exists(os.path.join(test_save))):
        os.mkdir(os.path.join(test_save))
    
    # 每一个类别文件下的所有文件名，存到name中
    name = os.listdir(epath)
    for i in range(0, train_nums):
        shutil.copy(os.path.join(epath,name[i]),os.path.join(train_save,name[i]))
    for i in range(train_nums, train_nums+val_nums):    
        shutil.copy(os.path.join(epath,name[i]),os.path.join(val_save,name[i]))
    for i in range(train_nums+val_nums, e_nums):    
        shutil.copy(os.path.join(epath,name[i]),os.path.join(test_save,name[i]))
print('Done.')