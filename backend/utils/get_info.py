

# 拿到类别的中文名称
def get_names(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    names = []
    indexs = []
    for data in class_names:
        name_cn, name_en, index = data.split('->')  # name_cn为中文类别名，name_en为英文类别名
        names.append(name_cn)
        indexs.append(int(index))
        
    return names, indexs