import time
import collections

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.simplefilter('ignore')

from sklearn.preprocessing import scale
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# Display up to 150 rows and columns
pd.set_option('display.max_rows', 220)
pd.set_option('display.max_columns', 150)

# Set the figure size for plots
mpl.rcParams['figure.figsize'] = (14.6, 9.0)

# Set the Seaborn default style for plots
sns.set()

# Set the color palette
sns.set_palette(sns.color_palette("Blues"))

# Load the preprocessed GTD dataset
gtd_df = pd.read_csv('./data/gtd_eda_98to17.csv', low_memory=False, index_col = 0,
                      na_values=[''])

# unknown_df = pd.read_csv('./data/gtd_eda_98to17.csv', low_memory=False, index_col = 0,
#                       na_values=[''])

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

# Necessary for label encoding below
gtd_df['gname'] = gtd_df['gname'].astype('str')

# gtd_df.info(verbose=True)


#######################################################################
# Find the Major Groups
# Get the list of terrorist groups that have 20 or more attacks.

# Calculate the number of attacks by group
groups = gtd_df['gname'].value_counts()

# Include groups with at least 20 attacks
groups = groups[groups > 19]

# Exclude unknown groups
#group_list = groups.index[groups.index != 'Unknown']
group_list = groups.index

# Subset the data to major groups
major_groups = gtd_df[gtd_df['gname'].isin(group_list)]

# Display the number of attacks by group
# print(major_groups['gname'].value_counts())

#######################################################################################
# Drop Text and Datetime Attributes
# 去除不需要的属性
# Remove the text and datetime attributes, which will not be used in the models.

major_groups = major_groups.drop(['provstate', 'city', 'summary', 'corp1', 'target1',
                                  'scite1', 'dbsource', 'incident_date'], axis=1)

# print(major_groups.info(verbose = True))

#######################################################################################
# Standardize the Numeric Attributes
scaler = preprocessing.RobustScaler()

# List of numeric attributes
scale_attrs = ['nperpcap', 'nkill', 'nkillus', 'nkillter', 'nwound', 'nwoundus', 'nwoundte']

# Standardize the attributes in place
# 数据规范化
major_groups[scale_attrs] = scaler.fit_transform(major_groups[scale_attrs])

# View the transformation
# print(major_groups[scale_attrs].describe().transpose())

##########################################################################################
# Separate Known and Unknown
# Excluded Unknown groups

known_maj_groups = major_groups[gtd_df['gname'] != "Unknown"]
print("Known Major Groups: {}".format(known_maj_groups.shape))

# Only include Unknown groups
unknown_maj_groups = major_groups[gtd_df['gname'] == "Unknown"]
print("Unknown Major Groups: {}".format(unknown_maj_groups.shape))
# Known Major Groups: (49852, 40)
# Unknown Major Groups: (59049, 40)


##############################################################################################
# Encode the Target Attribute
# 将恐怖组织的文本值转换为随机森林模型的编码数值。
# Create the encoder
le = preprocessing.LabelEncoder()

# Fit the encoder to the target
le.fit(known_maj_groups['gname'])

# View the labels
# print(list(le.classes_))

# View the encoded values for th terrorist group names
label_codes = le.transform(known_maj_groups['gname'])
print(label_codes)

# Convert some integers into their category names
# print(list(le.inverse_transform([0, 1, 2, 27])))
# output: ['Abdullah Azzam Brigades', 'Abu Sayyaf Group (ASG)',
# 'Adan-Abyan Province of the Islamic State', 'Anti-Semitic extremists']



#####################################################################################
# 创建训练数据集
# Create Training and Testing Datasets
# Seed for reproducible results
seed = 1009

# Predictor variables
X = pd.get_dummies(known_maj_groups.drop(['gname'], axis=1), drop_first=True)

# Labels
y = label_codes # 有名称的组织

# Create an 80/20 split for training and testing data
# 训练使用有名称的组织来训练
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                    random_state = seed, stratify = y)

#########################################################################################
# 创建机器学习模型——随机森林
# Random Forest Model
# Create the model using 1000 estimators.
start = time.time()

# Create the model
rf1 = RandomForestClassifier(n_estimators = 1000, oob_score = True, n_jobs = -1, random_state = seed)

# Fit it to the training data
rf1.fit(X_train, y_train)

end = time.time()
print("Execution Seconds: {}".format((end - start)))
print("\n")
print(rf1)
#########################################################################################
# 随机森林里重要的特征，前25个重要的特征
# 不用随机森林，则删除这里
# Feature Importance
# Display the top 25 features by importance in descending order.
# Get the modified column names with one hot encoding
column_names = list(X_train.columns.values)

# Create a descending sorted list of variables by feature importance
var_imp = sorted(zip(map(lambda x: x, rf1.feature_importances_), column_names),
             reverse = True)

print("\nFeatures Ranking - Top 25:\n")
for f in var_imp[0:25]:
    print(f)
#####################################################################################################
# 随机森林中，获取特征，标准偏差和下标
# 在画图，画前25个重要的特征
# Get the features, standard deviation and indices
importances = rf1.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf1.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

# Filter to the top 25
top_n = 25
top_n_importances = importances[indices][0:top_n]
top_n_std = std[0:top_n]
top_n_indices = indices[0:top_n]

# Create a list of x labels
x_labels = [column_names[t] for t in top_n_indices]

# Plot the results
plt.figure()
plt.title("Feature Importance - Top 25",  fontsize=20)
plt.bar(range(top_n), top_n_importances, color="firebrick", yerr = top_n_std, align="center")
plt.xticks(rotation=90)
plt.xticks(range(top_n), x_labels)
plt.xlim([-1, top_n])
plt.show()

############################################################################################
# 利用训练好的随机森林模型验证，验证集，有组织的
# Predict labels on the test dataset
pred_labels1 = rf1.predict(X_test)

# Calculate the accuracy of the model
acc_score1 = accuracy_score(y_test, pred_labels1)
print("\nAccuracy: {}".format(acc_score1))

# Calculate the precision of the model
prec_score1 = precision_score(y_test, pred_labels1, average='weighted')
print("\nPrecision: {}".format(prec_score1))

# Calculate the recall of the model
rcll_score1 = recall_score(y_test, pred_labels1, average='weighted')
print("\nRecall: {}".format(rcll_score1))

# Calculate the F1 of the model
f1_score1 = f1_score(y_test, pred_labels1, average='weighted')
print("\nF1: {}".format(f1_score1))

#########################################################################################
# 用没有组织的数据集进行预测
# Apply Model to Uknown Groups
# Of the 106,544 observations, 56.25% have a group classification of Unknown.
# Predictor variables
X_unknown = pd.get_dummies(unknown_maj_groups.drop(['gname'], axis=1), drop_first=True)


# Predict labels on the unknown dataset
pred_labels2 = rf1.predict(X_unknown)

# Get the list of predicted labels for unknown observations
unknown_preds = list(le.inverse_transform(pred_labels2))

# Calculate the counts for each group
unknown_counts = collections.Counter(unknown_preds)

# Top 5 Predicted Unknown Groups
# The top 5 predicted groups account for 97.59% of the unknown observations.

# Top 5 unknown
unknown_top5 = pd.DataFrame(unknown_counts.most_common()[0:25], columns=['Group', 'Attacks'])

cumsum = 0

# Display the top 25 groups with counts
for index, row in unknown_top5.iterrows():
    print("{} : {}".format(row['Group'], row['Attacks']))
    cumsum += row['Attacks']


print("\nTop 25 Account For: {}%".format((cumsum / X_unknown.shape[0])*100))

# Muslim Extremists 伊斯兰极端主义
# Almost half of the predicted unknown group observations are Muslim extremists.
# Set the color palette in reverse
colors = sns.color_palette('Blues', len(unknown_top5))
colors.reverse()
plt.figure(figsize=(14.6, 9.0))

# Plot bar chart with index as y values
ax = sns.barplot(unknown_top5.Attacks, unknown_top5.index, orient='h', palette=colors)
ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

# Reset the y labels
ax.set_yticklabels(unknown_top5.Group)
ax.set_xlabel(xlabel='Number of Attacks', fontsize=16)
ax.set_title(label='Top 25 Predicted Groups', fontsize=20)
plt.show();

"""
Accuracy: 0.9077324240296861

Precision: 0.8986471243084373

Recall: 0.9077324240296861

F1: 0.8975870813015374
"""

