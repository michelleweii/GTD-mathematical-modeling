import time
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Display up to 150 rows and columns
pd.set_option('display.max_rows', 150)
pd.set_option('display.max_columns', 150)

# Set the figure size for plots
mpl.rcParams['figure.figsize'] = (14.6, 9.0)

# Set the Seaborn default style for plots
sns.set()

# Set the color palette
sns.set_palette(sns.color_palette("muted"))

# Load the preprocessed GTD dataset
gtd_df = pd.read_csv('./data/gtd_eda_98to17.csv', low_memory=False, index_col = 0,
                      na_values=[''])


# Convert Attributes to Correct Data Type
# List of attributes that are categorical
cat_attrs = ['extended_txt', 'country_txt', 'region_txt', 'specificity', 'vicinity_txt',
             'crit1_txt', 'crit2_txt', 'crit3_txt', 'doubtterr_txt', 'multiple_txt',
             'success_txt', 'suicide_txt', 'attacktype1_txt', 'targtype1_txt',
             'targsubtype1_txt', 'natlty1_txt', 'guncertain1_txt', 'individual_txt',
             'claimed_txt', 'weaptype1_txt', 'weapsubtype1_txt', 'property_txt',
             'ishostkid_txt', 'INT_LOG_txt', 'INT_IDEO_txt', 'INT_MISC_txt', 'INT_ANY_txt']

for cat in cat_attrs:
    gtd_df[cat] = gtd_df[cat].astype('category')

# Data time feature added during EDA
gtd_df['incident_date'] = pd.to_datetime(gtd_df['incident_date'])

# To prevent a mixed data type
gtd_df['gname'] = gtd_df['gname'].astype('str')

# gtd_df.info(verbose=True)

# Create Training and Testing Datasets
# Seed for reproducible results
seed = 1009

# Predictor variables with one hot encoding
X = pd.get_dummies(gtd_df[['country_txt', 'region_txt', 'attacktype1_txt', 'nkill','nwound']],
                   drop_first = True)

# Labels
y = gtd_df['weaptype1_txt']

# Create an 80/20 split for training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = seed, stratify = y)

print(X_train.shape)
print(y_train.shape)
# neural network
modelfile = './kerasmodel/net.model'
outputfile = './kerasmodel/value.xlsx'

from keras.models import Sequential
from keras.layers.core import Dense,Activation


model = Sequential()  # 建立模型
model.add(Dense(5,32))
model.add(Activation('relu')) # 用relu函数作为激活函数
model.add(Dense(32,13))
model.compile(loss='categorical_crossentropy',optimizer='adam') # 编译模型
model.fit(X_train,y_train,nb_epoch=10000,batch_size=16) # 训练模型
# model.save_weights(modelfile) # 保存模型参数

# 预测
y_pred = model.predict_classes(X_test).reshape(len(y_test))

from sklearn.metrics import confusion_matrix

confusion_mat = confusion_matrix(y_test,y_pred)

print(confusion_mat)

def plot_confusion_matrix(confusion_mat):
    plt.imshow(confusion_mat,interpolation='nearest',cmap=plt.cm.Paired)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks=np.arange(4)
    plt.xticks(tick_marks,tick_marks)
    plt.yticks(tick_marks,tick_marks)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

plot_confusion_matrix(confusion_mat)


# CART决策树
from sklearn.tree import DecisionTreeClassifier

treefile = './tree/tree.pkl'

tree = DecisionTreeClassifier()
tree.fit(X_train,y_train)



# 保存模型
from sklearn.externals import joblib
joblib.dump(tree,treefile)

"""
# 预测
y_pred = tree.predict_classes(X_test).reshape(len(y_test))
# Calculate the accuracy
score_cart = accuracy_score(y_test, y_pred)
print("\nAccuracy: {}".format(score_cart))
"""

from sklearn.metrics import confusion_matrix

confusion_mat = confusion_matrix(y_test,y_pred)

print(confusion_mat)

def plot_confusion_matrix(confusion_mat):
    plt.imshow(confusion_mat,interpolation='nearest',cmap=plt.cm.Paired)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks=np.arange(4)
    plt.xticks(tick_marks,tick_marks)
    plt.yticks(tick_marks,tick_marks)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

plot_confusion_matrix(confusion_mat)

# 绘制决策树模型的roc曲线
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test,tree.predict_proba(X_test)[:,1],pos_label=1)

plt.plot(fpr,tpr,linewidth=2,label="ROC OF CART")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.ylim(0,1.05)
plt.xlim(0,1.05)
plt.legend(loc=4)
plt.show()