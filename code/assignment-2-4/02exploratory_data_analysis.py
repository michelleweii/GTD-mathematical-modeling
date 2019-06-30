import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Configure notebook output
#from IPython.core.interactiveshell import InteractiveShell
#InteractiveShell.ast_node_interactivity = "all"

# Display up to 150 rows and columns
pd.set_option('display.max_rows', 150)
pd.set_option('display.max_columns', 150)

# Set the figure size for plots
mpl.rcParams['figure.figsize'] = (12, 7)

# Set the Seaborn default style for plots
sns.set()

# Set the color palette
sns.set_palette(sns.color_palette("muted"))

# Load the preprocessed GTD dataset
gtd_df = pd.read_csv('./data/gtd_preprocessed_98to17.csv', low_memory=False, index_col = 0,
                      na_values=[''])


gtd_df.loc[gtd_df['weaptype1_txt'] ==
           'Vehicle (not to include vehicle-borne explosives, i.e., car or truck bombs)',
           'weaptype1_txt'] = 'Vehicle (non-explosives)'

gtd_df.loc[gtd_df['attacktype1_txt'] ==
           'Hostage Taking (Barricade Incident)',
           'attacktype1_txt'] = 'Hostage Taking (Barricade)'

# List of attributes that are categorical
cat_attrs = ['extended_txt', 'country_txt', 'region_txt', 'specificity', 'vicinity_txt',
             'crit1_txt', 'crit2_txt', 'crit3_txt', 'doubtterr_txt', 'multiple_txt',
             'success_txt', 'suicide_txt', 'attacktype1_txt', 'targtype1_txt',
             'targsubtype1_txt', 'natlty1_txt', 'guncertain1_txt', 'individual_txt',
             'claimed_txt', 'weaptype1_txt', 'weapsubtype1_txt', 'property_txt',
             'ishostkid_txt', 'INT_LOG_txt', 'INT_IDEO_txt', 'INT_MISC_txt', 'INT_ANY_txt']

for cat in cat_attrs:
    gtd_df[cat] = gtd_df[cat].astype('category')

gtd_df.info(verbose=True)


gtd_df[['nperpcap', 'nkill', 'nkillus', 'nkillter', 'nwound',
        'nwoundus', 'nwoundte']].dropna().describe(
    percentiles = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]).transpose()


# Function to impute either the median or mean
# 用于归因中位数或均值的函数
def fill_value(attr):
    fill = 0.0
    threshold = 3
    attr_clean = attr.dropna()
    attr_std = attr_clean.std()
    outliers = attr_clean[attr_clean > (threshold * attr_std)]

    if (outliers.count() > 0):
        fill = attr_clean.median()
    else:
        fill = attr_clean.mean()

    return fill

# Impute each of the numeric attributes that contain missing values
gtd_df['nperpcap'] = gtd_df['nperpcap'].fillna(fill_value(gtd_df['nperpcap']))
gtd_df['nkill'] = gtd_df['nkill'].fillna(fill_value(gtd_df['nkill']))
gtd_df['nkillus'] = gtd_df['nkillus'].fillna(fill_value(gtd_df['nkillus']))
gtd_df['nkillter'] = gtd_df['nkillter'].fillna(fill_value(gtd_df['nkillter']))
gtd_df['nwound'] = gtd_df['nwound'].fillna(fill_value(gtd_df['nwound']))
gtd_df['nwoundus'] = gtd_df['nwoundus'].fillna(fill_value(gtd_df['nwoundus']))
gtd_df['nwoundte'] = gtd_df['nwoundte'].fillna(fill_value(gtd_df['nwoundte']))

gtd_df[['nperpcap', 'nkill', 'nkillus', 'nkillter', 'nwound',
        'nwoundus', 'nwoundte']].describe(
    percentiles=[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]).transpose()

# Select the observations that contain null
ll_df = gtd_df[np.isnan(gtd_df.latitude)]
print(ll_df.shape)

# Chech how many observations have city set to Unknown
city_df = ll_df[(ll_df['city'] == "UNKNOWN")]
print(city_df['city'].value_counts())

# Remove observations containing missing missing values for latitude and longitude
gtd_clean = gtd_df.dropna().copy()
gtd_clean.info(verbose = True)

# 297 iday attributes contain 0 to represent unknown, setting 1
gtd_clean.loc[gtd_clean['iday'] == 0, 'iday'] = 1

gtd_clean['incident_date'] = (gtd_clean['iyear'].astype(str) + '-' +
                              gtd_clean['imonth'].astype(str) + '-' +
                              gtd_clean['iday'].astype(str))

gtd_clean['incident_date'] = pd.to_datetime(gtd_clean['incident_date'],
                                            format="%Y-%m-%d")
gtd_clean.info(verbose = True)

gtd_clean.to_csv("./data/gtd_eda_98to17.csv", sep = ",")

# Make a range of years to show categories with no observations
years = np.arange(1998, 2017)

# Draw a count plot to show the number of attacks each year
plt1 = sns.factorplot("iyear", data=gtd_clean, kind="count", color='steelblue', size=7.6, aspect=1.618)
plt1.set_xticklabels(step=3)
_ = plt.title('Attacks by Year', fontsize = 20)
_ = plt.xlabel('Year', fontsize = 16)
_ = plt.ylabel('Number of Attacks', fontsize = 16)
plt.show();

#Fatalities by Year
# Make a range of years to show categories with no observations
years = np.arange(1995, 2017)

df1 = gtd_clean[['iyear', 'nkill']]
gp1 = df1.groupby(['iyear'], as_index = False).sum()

# Draw a count plot to show the number of attacks each year
plt1 = sns.factorplot(x = 'iyear', y = 'nkill', data=gp1, kind = 'bar', color='steelblue', size=7.6, aspect=1.618)
plt1.set_xticklabels(step=3)
_ = plt.title('Fatalities by Year', fontsize = 20)
_ = plt.xlabel('Year', fontsize = 16)
_ = plt.ylabel('Number of Fatalities', fontsize = 16)
plt.show();



#Attack Locations
# 攻击地点
import folium

# Get a basic world map.
gtd_map = folium.Map(location=[30, 0], zoom_start=2);

# Take a sample of the data points
gtd_sample = gtd_clean.sample(3000);

# Draw markers on the map.
for index, row in gtd_sample.iterrows():
    folium.CircleMarker([row[7], row[8]], radius=0.5, color='Blues',
                        fill_color='#E74C3C').add_to(gtd_map);


# Show the map
gtd_map



# Attacks by Geographical Region
print('Attacks by Geographical Region')
print(gtd_clean.region_txt.value_counts())

# Attacks by Country
# 国家袭击
data = gtd_clean[['country_txt']].copy()
data['event_id'] = data.index

# Calculate the number of attacks
data = data.groupby(['country_txt']).agg(['count'])
data = data.reset_index()
data.columns = ['Country','Attacks']

# Order attacks descending
data = data.sort_values('Attacks', ascending=False)[0:20]
data = data.reset_index()

# Set the color palette in reverse
colors = sns.color_palette('Reds', len(data))
colors.reverse()
plt.figure(figsize=(12, 7))

# Plot bar chart with index as y values
ax = sns.barplot(data.Attacks, data.index, orient='h', palette=colors)
ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

# Reset the y labels
ax.set_yticklabels(data.Country)
ax.set_xlabel(xlabel='Number of Attacks', fontsize=16)
ax.set_title(label='Top 20 Countries by Total Attacks 1998 - 2017', fontsize=20)
plt.show();

# Attacks by Type
# 按类型攻击
ax = sns.factorplot('attacktype1_txt', data=gtd_clean, size=7.6, aspect=1.618, kind="count")
ax.set_xticklabels(rotation=60)
ax = plt.ylabel('Count', fontsize=16)
ax = plt.xlabel('',)
ax = plt.title('Total Attacks by Type 1998 - 2017', fontsize=20)

plt.show();

# Attacks by Weapon Type
# 武器类型的攻击
ax = sns.factorplot('weaptype1_txt', data=gtd_clean, size=7.6, aspect=1.618, kind="count")
ax.set_xticklabels(rotation=45)
ax = plt.ylabel('Count', fontsize=16)
ax = plt.xlabel('',)
ax = plt.title('Total Attacks by Weapon Type 1998 - 2017', fontsize=20)
plt.show();

# Fatalities Empirical Cumulative Distribution¶
# 死亡率经验累积分布
# Remove outliers
nkill_std = gtd_clean['nkill'].std()
nkill_no_outliers = gtd_clean[gtd_clean['nkill'] <= (3 * nkill_std)]

# x is the quantity measured
x = np.sort(nkill_no_outliers['nkill'])

# y is faction of data points that have value smaller than the corresponding x value
y = np.arange(1, len(x) + 1) / len(x)
_ = plt.plot(x, y, marker = '.', linestyle = 'none')
_ = plt.xlabel('Total Incident Fatalities')
_ = plt.ylabel('ECDF')
plt.margins(0.02)
plt.show()

# Ideological Attacks by Region
# 各地区的意识形态攻击
print('xtab 各地区的意识形态攻击')
xtab = pd.crosstab(index = gtd_clean['region_txt'], columns = gtd_clean['INT_IDEO_txt'], margins=True)
print(xtab)
