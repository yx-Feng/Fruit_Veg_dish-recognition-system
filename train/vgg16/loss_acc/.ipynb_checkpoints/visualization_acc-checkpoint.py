# accuracy可视化
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EPOCH = 60

# 读取csv中指定列的数据
data = pd.read_csv('./train_val.csv')
train_acc = data[['train_top1']]  # class 'pandas.core.frame.DataFrame'
val_acc = data[['val_top1']]
x = np.arange(1, EPOCH+1)
y1 = np.array(train_acc) # 将DataFrame类型转化为numpy数组
y2 = np.array(val_acc)

# 绘制train_val_acc图
plt.plot(x, y1, 'red', linewidth=2, label='Train Top1')
plt.plot(x, y2, 'blue', linewidth=2, label='Val Top1')
# x轴和y轴的刻度
plt.xlim(0, 62)
plt.ylim(40, 100)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('./train_val_acc.png')
plt.close()  # 关闭窗口