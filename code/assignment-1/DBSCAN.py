# -*- coding: utf-8 -*-

import numpy as np
from sklearn.cluster import FeatureAgglomeration,DBSCAN
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from openpyxl import Workbook
import matplotlib.colors
from sklearn import metrics
from sklearn.metrics import euclidean_distances
from sklearn.cluster import SpectralClustering

n_clusters = 5
gamma = 0.5

inputfile = './tmp/standardized.xlsx'
centersfile = './results/DBSCAN_centersfile.xlsx'
labelsfile = './results/DBSCAN_labelsfile.xlsx'


# 读取数据进行聚类分析
data = pd.read_excel(inputfile)
# 检查数据中是否有缺失值
print(np.isnan(data).any())
# 删除有缺失值的行
data.dropna(inplace=True)
# X=np.array(data)
# X=np.transpose(X)
# clf=FeatureAgglomeration(n_clusters=5,affinity='euclidean',memory='./tmp')
# clf = SpectralClustering(n_clusters=n_clusters,gamma=gamma)
# clf.fit(X)
# print(clf.)
# print(clf.labels_)
# 调用谱聚类进行聚类分析and预测
# data=data[0:50000]
# y_pred = SpectralClustering(gamma=gamma, n_clusters=n_clusters).fit_predict(data)
# print(y_pred)


# db是预测值
# 训练模型
db = DBSCAN(eps=0.3, min_samples=10).fit(data)
# core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
# core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

centroid = db.components_


print(centroid)
print("centroid_shape:")
print(centroid.shape)

print("Estimated number of clusters: %d" % n_clusters)
print("....")
print(labels)
"""
output:
5
....
[0 1 1 ... 2 1 1]

"""

labels = np.reshape(labels,(-1,1))
print("after reshape:")
print(labels)



#创建工作簿
wb = Workbook()
#创建表单
sheet = wb.active
sheet.title = "New Shit2"
#按i行j列顺序依次存入表格
for i in range(labels.shape[0]):
    for j in range(labels.shape[1]):
        sheet["A%d" % (i+1)].value = str(labels[i][j])
#保存文件
wb.save(labelsfile)


#创建工作簿
wb1 = Workbook()
#创建表单
sheet = wb1.active
sheet.title = "New Shit3"
#按i行j列顺序依次存入表格
for i in range(centroid.shape[0]):
    for j in range(centroid.shape[1]):
        sheet["A%d" % (i+1)].value = str(centroid[i][j])
#保存文件
wb.save(centersfile)