# 脚本作用：将所有文件目录修改为递增自然数
import os
import glob
 
path = "/hy-tmp/Food_Fruit_Veg"  # 根目录
file_name = sorted(os.listdir(path)) # 获取整个列表名称，排好序
file_num = len(file_name) #数组长度，int
 
i = 156 # 起始序号
 
for files in file_name:
    if(i<10):
        new_name = path + '/00' + str(i)
        i += 1
    elif(i<100):
        new_name = path + '/0' + str(i)
        i += 1
    else:
        new_name = path + '/' + str(i)
        i += 1
    
    os.rename(path+'/'+files,new_name)
 
print('Done.')