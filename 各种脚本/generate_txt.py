# 脚本作用：生成train.txt, val.txt, test.txt
import os
 
dataset_path='/hy-tmp/Fruits_Veg_Dishes/'  # 数据集存放的根目录
train_Dir = '/hy-tmp/Fruits_Veg_Dishes/train'
val_Dir = '/hy-tmp/Fruits_Veg_Dishes/val'
test_Dir = '/hy-tmp/Fruits_Veg_Dishes/test'
 
train_label=0
val_label=0
test_label=0
 
# os.listdir里面不是按实际目录的顺序组织的，需要排下序
train_list=sorted(os.listdir(train_Dir))
val_list=sorted(os.listdir(val_Dir))
test_list=sorted(os.listdir(test_Dir))
 
# 分别写入train.txt, val.txt, test.txt
with open('train.txt', 'w') as f1, open('val.txt', 'w') as f2, open('test.txt', 'w') as f3:
    for train_subDir in train_list:
        for train_filename in os.listdir(train_Dir+"/"+ train_subDir):
            f1.write(dataset_path + "train/" + train_subDir + "/" + train_filename + " " + str(train_label) + "\n")
        train_label+=1
 
    for val_subDir in val_list:
        for val_filename in os.listdir(val_Dir+"/"+ val_subDir):
            f2.write(dataset_path + "val/" + val_subDir + "/" + val_filename + " " + str(val_label) + "\n")
        val_label+=1
 
    for test_subDir in test_list:
        for test_filename in os.listdir(test_Dir+"/"+ test_subDir):
            f3.write(dataset_path + "test/" + test_subDir + "/" + test_filename + " " + str(test_label) + "\n")
        test_label+=1
 
print('Done.')