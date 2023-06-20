# 可视化
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EPOCH = 100

# 读取csv中指定列的数据
data = pd.read_csv('./train_acc.csv')
data_loss = data[['train_loss']]  # class 'pandas.core.frame.DataFrame'
data_acc = data[['train_top1']]
x = np.arange(1, EPOCH+1)
y1 = np.array(data_loss) # 将DataFrame类型转化为numpy数组
y2 = np.array(data_acc)
# 绘制loss-acc图
fig, ax1 = plt.subplots()
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.plot(x, y1, 'red', linewidth=2, label='Train loss')
ax1.grid(True)

ax2 = ax1.twinx()
ax2.set_ylabel('Acc')
ax2.plot(x, y2, 'blue', linewidth=2, label='Val acc')
fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.95))
fig.tight_layout()
plt.savefig('./loss-acc.png')