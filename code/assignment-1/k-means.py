# -*- coding: utf-8 -*-
# k-means聚类

import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from openpyxl import Workbook

inputfile = './tmp/standardized.xlsx'
# centersfile = '../results/centersfile.xlsx'
labelsfile = './results/labelsfile.xlsx'

k = 5

# 读取数据进行聚类分析
data = pd.read_excel(inputfile)
# 检查数据中是否有缺失值
print(np.isnan(data).any())
# 删除有缺失值的行
data.dropna(inplace=True)
# 调用kmeans进行聚类分析
kmodel = KMeans(n_clusters=k)
kmodel.fit(data) # 训练模型

print(kmodel.cluster_centers_) # 查看聚类中心
#print(kmodel.cluster_centers_.shape) # (5, 10)

# print(kmodel.cluster_centers_.shape) # (5, 10)
# print(kmodel.labels_) # 查看各样本对应的类别
# labels = kmodel.labels_
labels = np.reshape(kmodel.labels_,(-1,1))
# print(kmodel.labels_.shape) # (114182,)
print(labels)
#print(type(labels))
#print(labels.shape)

# labels=labels.astype(np.int32) #赋值操作后a的数据类型变化
# print(labels.dtype) #int32

# # writer=pd.ExcelWriter("test.xlsx")
# data_df = pd.DataFrame(labels, columns=["labels"]) # 将numpy转为pandas格式
# data_df.to_excel(labelsfile)
#print(data_df) # [114182 rows x 1 columns]

# f=open("aa.txt","a+")       # 以追加的方式
# for i in range(labels.shape[0]):
#      for j in range(labels.shape[1]):
#          f.write(str(labels[i][j]))
#          f.write("\n") 
#创建工作簿
wb = Workbook()
#创建表单
sheet = wb.active
sheet.title = "New Shit" 
#按i行j列顺序依次存入表格
for i in range(labels.shape[0]):
    for j in range(labels.shape[1]):
        sheet["A%d" % (i+1)].value = str(labels[i][j])
#保存文件
wb.save(labelsfile)
"""
[[4]
 [3]
 [3]
 ...
 [1]
 [0]
 [3]]
"""