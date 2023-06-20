import pymysql  # pip install pymysql
import xlrd  # 需要指定安装旧版本pip install xlrd==1.2.0

"""
一、连接mysql数据库
"""
# 打开数据库连接
conn = pymysql.connect(
    host='localhost',  # MySQL服务器地址
    user='root',  # 用户名
    password='123456',  # 密码
    charset='utf8',
    port=3306,  # 端口
    db='fruit_veg_dish',  # 数据库名称
)
# 使用cursor()方法获取操作游标
c = conn.cursor()

"""
二、读取excel文件
"""
FilePath = '../种类信息文件/info.xlsx'

# 1.打开excel文件
wkb = xlrd.open_workbook(FilePath)
# 2.获取sheet
sheet = wkb.sheet_by_index(0)  # 获取第一个sheet表
# 3.获取总行数
rows_number = sheet.nrows
# 4.遍历sheet表中所有行的数据，并保存至一个空列表cap[]
cap = []
for i in range(rows_number):
    x = sheet.row_values(i)  # 获取第i行的值（从0开始算起）
    cap.append(x)

# print(cap)
"""
三、将读取到的数据批量插入数据库
"""
for item in cap:
    no = int(item[0])
    category = item[1]
    description = item[2]
    # 格式化字符串，对sql进行赋值
    sql = "insert into info(`id`,`category`,`description`) value ({},'{}','{}')"
    c.execute(sql.format(no, category, description))
conn.commit()
conn.close()
print("插入数据完成！")
