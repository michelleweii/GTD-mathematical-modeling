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

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Display up to 150 rows and columns
pd.set_option('display.max_rows', 220)
pd.set_option('display.max_columns', 150)

# Set the figure size for plots
mpl.rcParams['figure.figsize'] = (14.6, 9.0)

# Set the Seaborn default style for plots
sns.set()

# Set the color palette
sns.set_palette(sns.color_palette("Reds"))

# Load the preprocessed GTD dataset
gtd_df = pd.read_csv('./data/gtd_eda_98to17.csv', low_memory=False, index_col = 0,
                      na_values=[''])

unknown_df = pd.read_csv('./data/gtd_eda_98to17.csv', low_memory=False, index_col = 0,
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

# Necessary for label encoding below
gtd_df['gname'] = gtd_df['gname'].astype('str')

# gtd_df.info(verbose=True)


##############################################################################################################
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

############################################################################################################
# Drop Text and Datetime Attributes
# 去除不需要的属性
# Remove the text and datetime attributes, which will not be used in the models.

major_groups = major_groups.drop(['provstate', 'city', 'summary', 'corp1', 'target1',
                                  'scite1', 'dbsource', 'incident_date'], axis=1)

# print(major_groups.info(verbose = True))

#############################################################################################################
# Standardize the Numeric Attributes
scaler = preprocessing.RobustScaler()

# List of numeric attributes
scale_attrs = ['nperpcap', 'nkill', 'nkillus', 'nkillter', 'nwound', 'nwoundus', 'nwoundte']

# Standardize the attributes in place
# 数据规范化
major_groups[scale_attrs] = scaler.fit_transform(major_groups[scale_attrs])

# View the transformation
# print(major_groups[scale_attrs].describe().transpose())

###############################################################################################################
# Separate Known and Unknown
# Excluded Unknown groups

known_maj_groups = major_groups[gtd_df['gname'] != "Unknown"]
print("Known Major Groups: {}".format(known_maj_groups.shape))

# Only include Unknown groups
unknown_maj_groups = major_groups[gtd_df['gname'] == "Unknown"]
print("Unknown Major Groups: {}".format(unknown_maj_groups.shape))
# Known Major Groups: (49852, 40)
# Unknown Major Groups: (59049, 40)


###############################################################################################################
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



#######################################################################################################
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

#####################################################################################################
# Setup arrays to store train and test accuracies
# knn
neighbors = np.arange(1, 12)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn2 = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier to the training data
    knn2.fit(X_train, y_train)

    # Compute accuracy on the training set
    train_accuracy[i] = knn2.score(X_train, y_train)

    # Compute accuracy on the testing set
    test_accuracy[i] = knn2.score(X_test, y_test)

# Compare the Training and Testing Scores
# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()

# # find the best k
knn3 = KNeighborsClassifier(n_neighbors = 3)
print("The KNN classifier parameter:\n")
print(knn3)
#
# # Fit it using the training data
knn3.fit(X_train, y_train)

x_eventid =
# # 返回预测属于某标签的概率
print(knn3.predict_proba(x_eventid))