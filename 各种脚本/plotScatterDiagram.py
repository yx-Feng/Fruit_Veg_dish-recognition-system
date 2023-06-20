# 绘制每个种类的样本数量的散点图
import matplotlib.pyplot as plt
import numpy as np
import os
 
images_path = '/hy-tmp/Fruits_Veg_Dishes/images/'
label=[]
sample_size=[]
for sub_dir in sorted(os.listdir(images_path)):
    label.append(sub_dir)
    sample_size.append(len(os.listdir(images_path+'/'+sub_dir)))
    if len(os.listdir(images_path+'/'+sub_dir)) <= 600:
        print(sub_dir)
 
plt.scatter(label,sample_size)
my_x_ticks = np.arange(0, 250, 50)  # 设置x轴刻度
my_y_ticks = np.arange(0, 1400, 200)  # 设置y轴刻度
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)
plt.xlabel('category index')
plt.ylabel('sample size')
plt.show()