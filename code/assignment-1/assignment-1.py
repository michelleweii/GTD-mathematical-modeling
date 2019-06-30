# -*- coding: utf-8 -*-
# 数据标准化到[0,1]
import pandas as pd

# 参数初始化
filename = '../data/gtd_assignment1.xlxs'
standgtdfile = '../tmp/standardized.xls'

data = pd.read_excel(filename, index_col= 'eventid') # 读取数据

data = (data-data.min())/(data.max()-data.min()) # 离差标准化
data = data.reset_index()

data.to_excel(standgtdfile, index=False) # 保存结果