# Predicting Terrorist Attacks
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
sns.set_palette(sns.color_palette("muted"))

# Load the preprocessed GTD dataset
gtd_df = pd.read_csv('./data/gtd_eda_15to17.csv', low_memory=False, index_col = 0,
                      na_values=[''])

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

# Necessary for single data type
gtd_df['gname'] = gtd_df['gname'].astype('str')

gtd_df.info(verbose=True)

gtd_df = gtd_df.drop(['provstate', 'city', 'summary', 'corp1', 'target1',
                                  'scite1', 'dbsource'], axis=1)

gtd_df.info(verbose = True)


scaler = preprocessing.StandardScaler()

# List of numeric attributes
scale_attrs = ['nperpcap', 'nkill', 'nkillus', 'nkillter', 'nwound', 'nwoundus', 'nwoundte']

# Normalize the attributes in place
gtd_df[scale_attrs] = scaler.fit_transform(gtd_df[scale_attrs])

# View the transformation
gtd_df[scale_attrs].describe().transpose()

scaler = preprocessing.StandardScaler()

# List of numeric attributes
scale_attrs = ['nperpcap', 'nkill', 'nkillus', 'nkillter', 'nwound', 'nwoundus', 'nwoundte']

# Normalize the attributes in place
gtd_df[scale_attrs] = scaler.fit_transform(gtd_df[scale_attrs])

# View the transformation
gtd_df[scale_attrs].describe().transpose()

iraq_df = gtd_df[gtd_df['country_txt'] == "Iraq"].query('2014<iyear<=2017')

iraq_df.info(verbose = True)


# Group by incident_date
iraq_counts = iraq_df.groupby(['incident_date'], as_index = False).count()

# Select incident_date and a column for the counts
iraq_counts = iraq_counts[['incident_date', 'iyear']]
iraq_counts.columns = ['incident_date','daily_attacks']
iraq_counts.head()

# Reindex and Fill
idx = pd.date_range('2015-01-01', '2017-12-31')

iraq_ts = iraq_counts.set_index('incident_date')

iraq_ts = iraq_ts.reindex(idx, fill_value=0)
iraq_ts.head()

print(iraq_ts.describe())



# Daily Plot - Total Attacks
iraq_ts.plot()
plt.title('Iraq Daily Attacks: 2015 - 2017', fontsize=20);
plt.ylabel('Total Per Day')
plt.show();

# Weekly average using the resampled daily data.
# Weekly Plot - Average Attacks
weekly_summary = pd.DataFrame()
weekly_summary['weekly_avg_attacks'] = iraq_ts.daily_attacks.resample('W').mean()

weekly_summary.plot()
plt.title('Iraq Weekly Attacks: 2015 - 2017', fontsize=20);
plt.ylabel('Average Per Week')
plt.show();

# Monthly Plot - Average Attacks
# Monthly average using the resampled daily data.

monthly_summary = pd.DataFrame()
monthly_summary['monthly_avg_attacks'] = iraq_ts.daily_attacks.resample('M').mean()

monthly_summary.plot()
plt.title('Iraq Monthly Attacks: 2015 - 2017', fontsize=20);
plt.ylabel('Average Per Month')
plt.show();

# Exponential Weighted Moving Average
# Apply smoothing using exponential weighted moving average.
# Use a 30 day span for averaging
iraq_ewm = iraq_ts.ewm(span=30, adjust=False).mean()

iraq_ewm.head()

# Exponential Weighted Moving Average¶
# Daily attacks in Iraq, 2015 to 2017.
iraq_ewm.plot()
plt.title('Iraq Daily Attacks: 2015 - 2017', fontsize=20);
plt.ylabel('Exponential Weighted Moving Average')
plt.show();

# Facebook Prophet
# Create a modified dataset to comply with the Facebook Prophet requirements.
import fbprophet

from fbprophet import Prophet
prophet = Prophet()

iraq_fb = iraq_ts.copy()
iraq_fb['index1'] = iraq_fb.index
iraq_fb.columns = ['y', 'ds']

print(iraq_fb.head())

# Iraq Holidays
# Iraq regional and national holidays covering 2015 - 2017.
# Load the preprocessed GTD dataset

iraq_holidays = pd.read_csv('./data/iraq-holidays.csv', na_values=[''])
print(iraq_holidays.head()) # 只显示4条数据

# Time Series Model
# Create the time serie model for attacks in Iraq and factor in Iraq holidays.
# Make the prophet model and fit on the data
prophet1 = fbprophet.Prophet(changepoint_prior_scale=0.15, holidays=iraq_holidays)
prophet1.fit(iraq_fb)


# Predict 1 Year of Future Dates
# Predict 365 days after last the data point of 2016-12-31.
# Specify 365 days out to predict
future_data = prophet1.make_future_dataframe(periods=365, freq = 'D')
future_data.tail()
# Predict the values
forecast_data = prophet1.predict(future_data)
print(forecast_data[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
#forecast_data.tail()

# Plot the Predictions
prophet1.plot(forecast_data, xlabel = 'Date', ylabel = 'Attacks')
plt.title('Predicted Terrorist Attacks in Iraq', fontsize=20);
plt.show();

# Examine Seasonality and Trend Components
# The spike in June occurs before Eid ul-Fitr or Id-Ul-Fitr (End of Ramadan), and the jump in December follows the Mouloud (Birth of the Prophet). The drop in September occurs near the Eid al-Adha (Feast of the Sacrifice) and the Islamic New Year. The low number of attacks on Friday corresponds to Muslims Friday prayer.
#
# Since Prophet uses an additive model, the y-axis represents the absolute change relative to the trend (Letham, 2018).
prophet1.plot_components(forecast_data)

# Zoom in to the Last Year Plus One Year of Predictions
# The forecasted trend seems to align with the yearly seasonality shown in the component analysis.
# 放大去年加上一年的预测
# ＃预测趋势似乎与成分分析中显示的年度季节性一致。
prophet1.plot(forecast_data, xlabel = 'Date', ylabel = 'Attacks')
plt.title('Predicted Terrorist Attacks in Iraq', fontsize=20);
plt.xlim(pd.Timestamp('2017-01-01'), pd.Timestamp('2018-12-31'))
plt.show();