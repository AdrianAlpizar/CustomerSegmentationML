import altair
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import pandas as pd
import numpy as np
import missingno as msno


#Importing the Libraries
import numpy as np
import pandas as pd
import datetime
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt, numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering
from matplotlib.colors import ListedColormap
from sklearn import metrics
import warnings
import sys, io
from annotated_text import annotated_text
import plotly.tools

from st_pages import Page, show_pages, add_page_title

# Specify what pages should be shown in the sidebar, and what their titles and icons
# should be
show_pages(
    [
        Page("project.py", "Customer Segmentation Research"),
        Page("project2.py", "Building Prediction Models"),
    ]
)

# Optional -- adds the title and icon to the current page
add_page_title()

#st.title('Customer Segmentation Research')


st.markdown("Customer segmentation is the process of dividing a company's customer base into distinct groups or segments based on certain characteristics, behaviors, or attributes they share. The goal of customer segmentation is to better understand the diverse needs, preferences, and behaviors of different customer groups in order to tailor marketing strategies, products, and services to effectively meet their specific requirements.")
st.caption('Customer Dataset')
df = pd.read_csv('marketing_campaign.csv', sep="\t")
st.dataframe(df)  # Same as st.write(df)


code = '''
    df.info()
    '''
st.code(code, language='python')

buffer = io.StringIO()
df.info(buf=buffer)
s = buffer.getvalue()
st.text(s)

st.markdown('This includes the "Matrix" visualization, which displays the entire dataset as a grid, highlighting missing values with white lines. This matrix form enables you to identify patterns and clusters of missingness, which could offer insights into potential relationships between missing values.')


code = '''
    # visualize missing data
    msno.matrix(df)
    '''
st.code(code, language='python')

fig = msno.matrix(df)
fig_copy = fig.get_figure()
fig_copy.savefig('plot.png', bbox_inches = 'tight')
st.image('plot.png')

code = '''
#return a dataframe illustrating the exact amount vs the % of missing data
def check_missing_percentage(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total Missing', 'Percent Missing'])
    return missing_data.head(20)
check_missing_percentage(df)
    '''
st.code(code, language='python')

#missing data
def check_missing_percentage(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total Missing', 'Percent Missing'])
    return missing_data.head(20)

st.dataframe(check_missing_percentage(df))

st.markdown(
    "As you can see only 24 missing values, less than 1%. Let's make a temporary dataframe to see all 24 rows which have the missing values:"
)

code = '''
df[df.isnull().any(axis=1)]
    '''
st.code(code, language='python')

st.dataframe(df[df.isnull().any(axis=1)])

st.markdown(
    "Many ways we can deal with these few missing data points. Option #1 could be to fill NaNs with the median of the 'Income Column, another way is to impute the values using KNNImputer. Option #2 is more exciting, but we will keep it simple and use the medial fill."
)



# make a copy of the original dataframe to test both imputed and mean fill options
imputed_df = df.copy()
mean_df = df.copy()

# get the index of the missing rows as a list
list(df[df.isnull().any(axis=1)].index)

# Fill using median
mean_df['Income']=mean_df['Income'].fillna(mean_df['Income'].median())

# Fill using KNNImputer
from sklearn.impute import KNNImputer
knn = KNNImputer(n_neighbors = 3)
knn.fit(imputed_df[imputed_df.columns.difference(['Education', 'Marital_Status','Dt_Customer'])])
X = knn.transform(imputed_df[imputed_df.columns.difference(['Education', 'Marital_Status','Dt_Customer'])])
imputed_df = pd.DataFrame(X, columns = imputed_df[imputed_df.columns.difference(['Education', 'Marital_Status','Dt_Customer'])].columns)

# compare results for both options
mean_v_imputed = pd.concat([mean_df.loc[list(df[df.isnull().any(axis=1)].index)]['Income'], imputed_df.loc[list(df[df.isnull().any(axis=1)].index)]['Income']], axis=1)
mean_v_imputed.columns=['Mean','Imputed']

code = '''
# make a copy of the original dataframe to test both imputed and mean fill options
imputed_df = df.copy()
mean_df = df.copy()

# get the index of the missing rows as a list
list(df[df.isnull().any(axis=1)].index)

# Fill using median
mean_df['Income']=mean_df['Income'].fillna(mean_df['Income'].median())

# Fill using KNNImputer
from sklearn.impute import KNNImputer
knn = KNNImputer(n_neighbors = 3)
knn.fit(imputed_df[imputed_df.columns.difference(['Education', 'Marital_Status','Dt_Customer'])])
X = knn.transform(imputed_df[imputed_df.columns.difference(['Education', 'Marital_Status','Dt_Customer'])])
imputed_df = pd.DataFrame(X, columns = imputed_df[imputed_df.columns.difference(['Education', 'Marital_Status','Dt_Customer'])].columns)

# compare results for both options
mean_v_imputed = pd.concat([mean_df.loc[list(df[df.isnull().any(axis=1)].index)]['Income'], imputed_df.loc[list(df[df.isnull().any(axis=1)].index)]['Income']], axis=1)
mean_v_imputed.columns=['Mean','Imputed']
'''
st.code(code, language='python')
st.dataframe(mean_v_imputed)


st.markdown(
    "We can use the median results for now, we can always test our final model against both options and select the best performing."
)

code = '''
# Fill original dataframe using median
df['Income']=df['Income'].fillna(df['Income'].median())

# check that all mising data has been imputed
msno.matrix(df)
'''
st.code(code, language='python')
df['Income']=df['Income'].fillna(df['Income'].median())
msno.matrix(df)
fig = msno.matrix(df)
fig_copy = fig.get_figure()
fig_copy.savefig('plot.png', bbox_inches = 'tight')
st.image('plot.png')


st.markdown(
    """ 

Constant features, also known as constant columns, are features in a dataset that have the same value for all instances or examples. These features typically do not bring any value to a machine learning model for several reasons:

1. No Discriminatory Power: Since the feature has the same value for all instances, it cannot discriminate or differentiate between different examples. In other words, it does not contain any information that helps the model distinguish between different classes or predict the target variable.

2. Variance is Zero: The variance of a constant feature is zero, which means there is no variability in the data for that feature. Machine learning algorithms often rely on variability in features to make predictions and learn patterns. A constant feature lacks this variability, rendering it ineffective for learning meaningful relationships.

3. Doesn't Contribute to Model Generalization: Machine learning models aim to generalize patterns from the training data to unseen data. Constant features do not contribute to this generalization process, as they provide the same value regardless of the input. Including them can lead to overfitting, where the model learns noise in the data rather than true underlying patterns.

4. Computational Overhead: Including constant features in your dataset increases the computational overhead without providing any meaningful information. This can slow down training and prediction times, especially for algorithms that rely on matrix operations.

5. Potential to Degrade Model Performance: Depending on the specific algorithm and implementation, including constant features may negatively impact model performance. For example, some algorithms like k-means clustering can be heavily influenced by constant features, leading to suboptimal cluster assignments.

6. Dimensionality Reduction and Interpretability: Constant features add to the dimensionality of the data without contributing to the quality of the model. High-dimensional data can lead to increased computation time and reduced interpretability of the model's results.

In practice, it's a good practice to identify and remove constant features from your dataset before training a machine learning model. Removing such features can streamline the learning process, reduce noise, and improve the model's ability to focus on relevant patterns in the data.

    """
)


code = '''
# Check for constant features
pd.DataFrame(df.nunique()).sort_values(0).rename( {0: 'Unique Values'}, axis=1)
'''
st.code(code, language='python')

st.dataframe(pd.DataFrame(df.nunique()).sort_values(0).rename( {0: 'Unique Values'}, axis=1))

st.markdown(
'''
From the analysis completed, we can summerize the following:
- Z_Revenue & Z_CostContact have Constant values, no infomration to be learned from constant values, drop them.
- Response - AcceptedCmp5 are all Binary Variables.
- Marital_Status & Education are Categorical Variable.
- Kidhome & Teenhome are Discrete Ordinal Variables.
- All other features are Continuous Ordinal Variables.
'''
)

code = '''
# drop constant features
df.drop(['Z_CostContact', 'Z_Revenue'], axis=1, inplace=True) 
'''
st.code(code, language='python')

df.drop(['Z_CostContact', 'Z_Revenue'], axis=1, inplace=True) 


st.header('Univariate Analysis of features')
st.markdown(
'''
Univariate analysis is a fundamental and essential step in the process of data analysis, especially when exploring and understanding a dataset. 
Univariate analysis helps you get a basic understanding of the distribution, central tendency, and spread of individual variables. 
This exploration can reveal patterns, trends, outliers, and potential data quality issues.
'''
)

code = '''
# plot Maritial Status & Education % levels from the dataset
cust_count=df.groupby("Marital_Status").count()['Year_Birth']
print(len(cust_count))
label=df.groupby('Marital_Status').count()['Year_Birth'].index
fig, ax = plt.subplots(1, 2, figsize = (10, 12))
ax[0].pie(cust_count, labels=label, shadow=True, autopct='%1.2f%%',radius=2,explode=[0.1,0.1,0.1,0.1,0.1, 0.1,0.1,0.1])
ax[0].set_title('Maritial Status', y=-0.6)

cust_count = df.groupby("Education").count()['Year_Birth']
print(len(cust_count))
label = df.groupby('Education').count()['Year_Birth'].index
ax[1].pie(cust_count, labels=label, shadow=True, autopct='%1.2f%%',radius=2,explode=[0.1,0.1,0.1,0.1,0.1])
ax[1].set_title('Education Level', y=-0.6)
plt.subplots_adjust(wspace = 1.5, hspace =0)
'''
st.code(code, language='python')

# Maritial Status & Education levels
cust_count=df.groupby("Marital_Status").count()['Year_Birth']
print(len(cust_count))
label=df.groupby('Marital_Status').count()['Year_Birth'].index
fig, ax = plt.subplots(1, 2, figsize = (10, 12))
ax[0].pie(cust_count, labels=label, shadow=True, autopct='%1.2f%%',radius=2,explode=[0.1,0.1,0.1,0.1,0.1, 0.1,0.1,0.1])
ax[0].set_title('Maritial Status', y=-0.6)

cust_count = df.groupby("Education").count()['Year_Birth']
print(len(cust_count))
label = df.groupby('Education').count()['Year_Birth'].index
ax[1].pie(cust_count, labels=label, shadow=True, autopct='%1.2f%%',radius=2,explode=[0.1,0.1,0.1,0.1,0.1])
ax[1].set_title('Education Level', y=-0.6)
plt.subplots_adjust(wspace = 1.5, hspace =0)
st.pyplot(fig)


st.subheader(
'''
Education Feature
'''
)

code = '''
# Check unique variables in Education
df['Education'].unique() 
fig = plt.figure()
plt.title('Customer Education Levels')
ax= sns.barplot(x=df['Education'].value_counts().index, y=df['Education'].value_counts())
'''
st.code(code, language='python')
st.write(df['Education'].unique())
#st.bar_chart(df['Education'].value_counts())
fig = plt.figure()
plt.title('Customer Education Levels')
ax= sns.barplot(x=df['Education'].value_counts().index, y=df['Education'].value_counts())
st.pyplot(fig)


st.markdown(
'''
Combining features to form fewer features in a dataset is known as feature engineering or dimensionality reduction. This process can offer several benefits and is often performed to improve the efficiency and effectiveness of data analysis and modeling. 
(PhD, 2nd Cycle, Graduation, and Masters) could be sumerized as "Graduate" Education, while (Basic) can be "Undergraduate" Education.
This will help avoid Multicollinearity, Overfitting, and reduce Dimensionality.
'''
)

code = '''
df['Education'] = df['Education'].replace(['PhD','2n Cycle','Graduation', 'Master'],'Post Graduate')  
df['Education'] = df['Education'].replace(['Basic'], 'Under Graduate')
fig = plt.figure()
plt.title('Customer Education Levels after dimensionality reduction')
ax= sns.barplot(x=df['Education'].value_counts().index, y=df['Education'].value_counts())
'''
st.code(code, language='python')

df['Education'] = df['Education'].replace(['PhD','2n Cycle','Graduation', 'Master'],'Post Graduate')  
df['Education'] = df['Education'].replace(['Basic'], 'Under Graduate')
#st.bar_chart(df['Education'].value_counts())
fig = plt.figure()
plt.title('Customer Education Levels after dimensionality reduction')
ax= sns.barplot(x=df['Education'].value_counts().index, y=df['Education'].value_counts())
st.pyplot(fig)



#--------------------------- Univariate Analysis of Marital Status  ----------------------------------------------------------------------#

st.subheader(
'''
Marital_Status Feature
'''
)

code = '''
# Check unique variables in Marital_Status
df['Marital_Status'].unique() 
'''
st.code(code, language='python')

st.write(df['Marital_Status'].unique())

code = '''
# plot customer's marital statuses
fig = plt.figure()
plt.title('Customers Marital Status')
ax= sns.barplot(x=df['Marital_Status'].value_counts().index, y=df['Marital_Status'].value_counts())
'''

#st.bar_chart(df['Marital_Status'].value_counts())
fig = plt.figure()
plt.title("Customers Marital Status")
ax= sns.barplot(x=df['Marital_Status'].value_counts().index, y=df['Marital_Status'].value_counts())
st.pyplot(fig)

code = '''
# reduce stauses down to relationship and single
df['Marital_Status'] = df['Marital_Status'].replace(['Married', 'Together'],'Relationship')
df['Marital_Status'] = df['Marital_Status'].replace(['Divorced', 'Widow', 'Alone', 'YOLO', 'Absurd'],'Single')

# plot the new status
fig = plt.figure()
plt.title('Customers Marital Status after Dimensionality Reduction')
ax= sns.barplot(x=df['Marital_Status'].value_counts().index, y=df['Marital_Status'].value_counts())
'''
st.code(code, language='python')

df['Marital_Status'] = df['Marital_Status'].replace(['Married', 'Together'],'Relationship')
df['Marital_Status'] = df['Marital_Status'].replace(['Divorced', 'Widow', 'Alone', 'YOLO', 'Absurd'],'Single')
#st.bar_chart(df['Marital_Status'].value_counts())
fig = plt.figure()
plt.title('Customers Marital Status after Dimensionality Reduction')
ax= sns.barplot(x=df['Marital_Status'].value_counts().index, y=df['Marital_Status'].value_counts())
st.pyplot(fig)


#--------------------------- Univariate Analysis of Year_Birth  ----------------------------------------------------------------------#

st.subheader(
'''
Year_Birth Feature
'''
)

code = '''
# visualize the distribution of when the current customers were born
fig = plt.figure(figsize=(20, 6))
plt.title('Age distribution')
ax = sns.histplot(df['Year_Birth'].sort_values(), bins=56)
sns.rugplot(data=df['Year_Birth'], height=.05)
plt.xticks(np.linspace(df['Year_Birth'].min(), df['Year_Birth'].max(), 56, dtype=int, endpoint = True))
'''
st.code(code, language='python')

# Age Range
fig = plt.figure(figsize=(20, 6))
plt.title('Age distribution')
ax = sns.histplot(df['Year_Birth'].sort_values(), bins=56)
sns.rugplot(data=df['Year_Birth'], height=.05)
plt.xticks(np.linspace(df['Year_Birth'].min(), df['Year_Birth'].max(), 56, dtype=int, endpoint = True))
st.pyplot(fig)

st.markdown(
'''
Having the birth year of the customers serves no learning value, instead we could feature engineer a new feature with the "age" of that customer.
This way we can perform analysis customer age for marketing strategies.
'''
)

code = '''
# subtract current year from the biurth year of the customer to get their age
df['Age'] = 2023 - df.Year_Birth.to_numpy()
df.drop('Year_Birth', axis=1, inplace=True)


# plot the customers ages as a distribution
fig = plt.figure(figsize=(20, 6))
plt.title('Age distribution')
ax = sns.histplot(df['Age'].sort_values(), bins=100)
sns.rugplot(data=df['Age'], height=.05)
plt.xticks(np.linspace(df['Age'].min(), df['Age'].max(), 56, dtype=int, endpoint = True))
'''
st.code(code, language='python')

# Change Year_Birth to Age (Age is more informative)
df['Age'] = 2023 - df.Year_Birth.to_numpy()
df.drop('Year_Birth', axis=1, inplace=True)


# Age Range
fig = plt.figure(figsize=(20, 6))
plt.title('Age distribution')
ax = sns.histplot(df['Age'].sort_values(), bins=100)
sns.rugplot(data=df['Age'], height=.05)
plt.xticks(np.linspace(df['Age'].min(), df['Age'].max(), 56, dtype=int, endpoint = True))
st.pyplot(fig)


st.markdown(
'''
Already we can asses the age feature has outliers. We will handle outliers later using a pipeline on the entire dataset.
'''
)

#--------------------------- Univariate Analysis of Kids  ----------------------------------------------------------------------#

st.subheader("Kidhome / Teenhome Feature")

code = '''
# plot how many kids vs teens in each household
fig = plt.figure(figsize=(15,5))
plt.subplot(121)
ax= sns.barplot(x=df['Kidhome'].value_counts().index, y=df['Kidhome'].value_counts())

plt.subplot(122)
ax= sns.barplot(x=df['Teenhome'].value_counts().index, y=df['Teenhome'].value_counts())
st.pyplot(fig)
'''
st.code(code, language='python')

# Kid Home & Teen Home
fig = plt.figure(figsize=(15,5))
plt.title('Kids v. Teen Houshold')
plt.subplot(121)
ax= sns.barplot(x=df['Kidhome'].value_counts().index, y=df['Kidhome'].value_counts())

plt.subplot(122)
ax= sns.barplot(x=df['Teenhome'].value_counts().index, y=df['Teenhome'].value_counts())
st.pyplot(fig)


st.markdown(
'''
We can conclude that most costumers have 0 kids. Also, we could signifacntly reduce the dimensionality of the Kidhome and Teenhome features by adding them into a "Kids" feature since teens and kids fall under the same category.
'''
)
code = '''
# plot how many kids vs teens in each household
df['Kids'] = df['Kidhome'] + df['Teenhome']

# plot the new Kids feature
fig = plt.figure(figsize=(20, 6))
plt.title('Kids distribution')
ax= sns.barplot(x=df['Kids'].value_counts().index, y=df['Kids'].value_counts())
'''
st.code(code, language='python')

df['Kids'] = df['Kidhome'] + df['Teenhome']
fig = plt.figure(figsize=(20, 6))
plt.title('Kids distribution')
ax= sns.barplot(x=df['Kids'].value_counts().index, y=df['Kids'].value_counts())
st.pyplot(fig)


#--------------------------- Univariate Analysis of Response (target Variable)  ----------------------------------------------------------------------#

st.subheader("Response Feature (Target variable)")

code = '''
# plot the new Response feature
fig = plt.figure(figsize=(20, 6))
ax= sns.barplot(x=df['Response'].value_counts().index, y=df['Response'].value_counts())
st.pyplot(fig)
'''
st.code(code, language='python')

fig = plt.figure(figsize=(20, 6))
ax= sns.barplot(x=df['Response'].value_counts().index, y=df['Response'].value_counts())
st.pyplot(fig)

st.markdown("""
The target variable, also known as the dependent variable or response variable, is the variable in a dataset that you are trying to predict or explain. In other words, it's the outcome or result you want to understand or model. The importance of the target variable cannot be understated, as it defines the purpose of your analysis or modeling task.
When the target variable is imbalanced, it means that the distribution of different classes or categories within the target variable is not roughly equal. One class might have significantly more instances than the other class(es).      
           """)



#--------------------------- Univariate Analysis of MntTotal ----------------------------------------------------------------------#

st.subheader("MntWines, MntFruits, MntMeatProducts, MntFishProducts, MntSweetProducts, MntGoldProds Features")

# Add 'MntTotal' - the total purchasing amount of all products
df['MntTotal'] = np.sum(df.filter(regex='Mnt'), axis=1)


code = '''
# add all of the products into a "total" feature
df['MntTotal'] = np.sum(df.filter(regex='Mnt'), axis=1)

# plot the MntTotal feature as a distribution & box plot
fig = plt.figure(figsize=(15,5))
plt.title('MntTotal distribution')
plt.subplot(121)
ax = sns.distplot(df["MntTotal"])

plt.title('MntTotal Box Plot')
plt.subplot(122)
ax = df["MntTotal"].plot.box()
'''
st.code(code, language='python')

fig = plt.figure(figsize=(15,5))
plt.title('MntTotal distribution')
plt.subplot(121)
sns.distplot(df["MntTotal"])

plt.title('MntTotal Box Plot')
plt.subplot(122)
df["MntTotal"].plot.box()
st.pyplot(fig)


st.markdown('Once again we can asses there are outliers in the MinTotal feature, this will be added to the pipeline.')

#--------------------------- Univariate Analysis of NumTotal  ----------------------------------------------------------------------#

st.subheader("NumDealsPurchases, NumWebPurchases, NumCatalogPurchases, NumStorePurchases, NumWebVisitsMonth Features")

# Add 'NumTotal' - the total purchasing number of different purchasing types
df['NumTotal'] = np.sum(df.filter(regex='Purchases'), axis=1)

code = '''
# add all of the purchases into a "total" feature
df['NumTotal'] = np.sum(df.filter(regex='Purchases'), axis=1)

# plot the NumTotal feature as a distribution & box plot
fig = plt.figure(figsize=(15,5))
plt.title('NumTotal distribution')
plt.subplot(121)
ax = sns.distplot(df["NumTotal"])

plt.title('NumTotal Box Plot')
plt.subplot(122)
ax = df["NumTotal"].plot.box()
'''
st.code(code, language='python')

fig = plt.figure(figsize=(15,5))
plt.title('NumTotal distribution')
plt.subplot(121)
ax = sns.distplot(df["NumTotal"])

plt.title('NumTotal Box Plot')
plt.subplot(122)
ax = df["NumTotal"].plot.box()
st.pyplot(fig)


#--------------------------- Univariate Analysis of TotalAccepted  ----------------------------------------------------------------------#

st.subheader("AcceptedCmp3, AcceptedCmp4, AcceptedCmp5, AcceptedCmp1, AcceptedCmp2 Features")

code = '''
# add all of the participated campaigns into a "total" feature
df['TotalAccepted'] = np.sum(df.filter(regex='Accepted'),axis=1)

# drop the individual campaigns
df.drop(columns=['AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1','AcceptedCmp2'],inplace=True)

# plot TotalAccepted
fig = plt.figure(figsize=(20, 6))
ax= sns.barplot(x=df['TotalAccepted'].value_counts().index, y=df['TotalAccepted'].value_counts())
st.pyplot(fig)
'''
st.code(code, language='python')

# We don't care the which compaign the customer participate in; Instead, we care about the total participation times
df['TotalAccepted'] = np.sum(df.filter(regex='Accepted'),axis=1)
df.drop(columns=['AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1','AcceptedCmp2'],inplace=True)
fig = plt.figure(figsize=(20, 6))
ax = sns.barplot(x=df['TotalAccepted'].value_counts().index, y=df['TotalAccepted'].value_counts())
st.pyplot(fig)


#--------------------------- Univariate Analysis of AvgWeb  ----------------------------------------------------------------------#

st.subheader("NumWebPurchases / NumWebVisitsMonth Features")

# Cacluate Average of purchases per visit to the website, which can reflect the personality of the customer.
df['AvgWeb'] = round(df['NumWebPurchases'] / df['NumWebVisitsMonth'], 2)
df.fillna({'AvgWeb' : 0},inplace=True) # Handling for cases where division by 0
df.replace(np.inf, 0, inplace=True)


code = '''
# Cacluate Average of purchases per website visit
df['AvgWeb'] = round(df['NumWebPurchases'] / df['NumWebVisitsMonth'], 2)
df.fillna({'AvgWeb' : 0},inplace=True) # Handling for cases where division by 0
df.replace(np.inf, 0, inplace=True)
'''
st.code(code, language='python')
st.dataframe(df['AvgWeb'])


code = '''
# plot AvgWeb distribution
fig = plt.figure(figsize=(8,8))
plt.title('AvgWeb distribution')
sns.distplot(df["AvgWeb"])
st.pyplot(fig)
'''
st.code(code, language='python')

fig = plt.figure(figsize=(8,8))
plt.title('AvgWeb distribution')
sns.distplot(df["AvgWeb"])
st.pyplot(fig)


#--------------------------- Univariate Analysis of Dt_customer  ----------------------------------------------------------------------#
st.subheader("Dt_customer Feature")

st.markdown("We have the date the costumer engaged with the company, although this is valuable, a more valuable verision of this statistics would be how long (in years) has this person has been a costumer.")
#CHANGING "Dt_customer" into timestamp format......

code = '''
# get current year and subtract the date person became costumner 
df['YearsACustomer'] = (pd.Timestamp('now').year) - (pd.to_datetime(df['Dt_Customer'], dayfirst=True).dt.year)

# plot YearsACustomer, which groups all costumers into a group of how long they have been a costumer
fig = plt.figure(figsize=(20, 6))
plt.title('Years being a customer')
ax= sns.barplot(x=df['YearsACustomer'].value_counts().index, y=df['YearsACustomer'].value_counts())
'''
st.code(code, language='python')

df['YearsACustomer'] = (pd.Timestamp('now').year) - (pd.to_datetime(df['Dt_Customer'], dayfirst=True).dt.year)
fig = plt.figure(figsize=(20, 6))
plt.title('Years being a customer')
ax= sns.barplot(x=df['YearsACustomer'].value_counts().index, y=df['YearsACustomer'].value_counts())
st.pyplot(fig)

#--------------------------- Univariate Analysis of Income  ----------------------------------------------------------------------#

st.subheader("Income Feature")

code = '''
# the usual, plot the distribution of income across costumers and box plot for outliers
fig = plt.figure(figsize=(15,5))
plt.subplot(121)
ax = sns.distplot(df["Income"])

plt.subplot(122)
ax = sns.boxplot(df["Income"])
'''
st.code(code, language='python')

fig = plt.figure(figsize=(15,5))
plt.subplot(121)
ax = sns.distplot(df["Income"])

plt.subplot(122)
ax = sns.boxplot(df["Income"])
st.pyplot(fig)

st.markdown('Another feature with outliers. Will add to pipeline as well.')







st.header('Bivariate Analysis')
#--------------------------- Bivariate Analysis of Education vs Target  ----------------------------------------------------------------------#
st.subheader('Bivariate Analysis of Education vs Target (Response)')

code = '''
# plot crosstab of education and response
pd.crosstab(df['Education'],df['Response'],margins=True).style.background_gradient(cmap='Greys')
fig = plt.figure(figsize=(20, 6))
plt.title('Years being a customer')
ax = sns.barplot(x="Education", y="Response", data=df)
'''
st.code(code, language='python')

pd.crosstab(df['Education'],df['Response'],margins=True).style.background_gradient(cmap='Greys')
fig = plt.figure(figsize=(20, 6))
plt.title('Years being a customer')
ax = sns.barplot(x="Education", y="Response", data=df)
st.pyplot(fig)


#--------------------------- Bivariate Analysis of Marital status vs Target  ----------------------------------------------------------------------#

st.subheader('Bivariate Analysis of Marital_Status vs Target (Response)')

code = '''
# plot crosstab of Marital_Status and response
pd.crosstab(df['Marital_Status'],df['Response'],margins=True).style.background_gradient(cmap='Greys')
fig = plt.figure(figsize=(20, 6))
plt.title('Years being a customer')
ax = sns.barplot(x="Marital_Status", y="Response", data=df)
'''
st.code(code, language='python')

pd.crosstab(df['Marital_Status'],df['Response'],margins=True).style.background_gradient(cmap='Greys')
fig = plt.figure(figsize=(20, 6))
plt.title('Years being a customer')
ax = sns.barplot(x="Marital_Status", y="Response", data=df)
st.pyplot(fig)

#--------------------------- Bivariate Analysis of Kids vs Target  ----------------------------------------------------------------------#

st.subheader('Bivariate Analysis of Kids vs Target (Response)')

code = '''
# plot crosstab of Kids and response
pd.crosstab(df['Kids'],df['Response'],margins=True).style.background_gradient(cmap='Greys')
fig = plt.figure(figsize=(20, 6))
plt.title('Years being a customer')
ax = sns.barplot(x="Kids", y="Response", data=df)
'''
st.code(code, language='python')

pd.crosstab(df['Kids'],df['Response'],margins=True).style.background_gradient(cmap='Greys')
fig = plt.figure(figsize=(20, 6))
plt.title('Years being a customer')
ax = sns.barplot(x="Kids", y="Response", data=df)
st.pyplot(fig)

#--------------------------- Bivariate Analysis of Total Accepted vs Target  ----------------------------------------------------------------------#
st.subheader('Bivariate Analysis of TotalAccepted vs Target (Response)')

code = '''
# plot crosstab of TotalAccepted and response
pd.crosstab(df['TotalAccepted'],df['Response'],margins=True).style.background_gradient(cmap='Greys')
fig = plt.figure(figsize=(20, 6))
plt.title('Years being a customer')
ax = sns.barplot(x="TotalAccepted", y="Response", data=df)
'''
st.code(code, language='python')

pd.crosstab(df['TotalAccepted'],df['Response'],margins=True).style.background_gradient(cmap='Greys')
fig = plt.figure(figsize=(20, 6))
plt.title('Years being a customer')
ax = sns.barplot(x="TotalAccepted", y="Response", data=df)
st.pyplot(fig)

#--------------------------- Bivariate Analysis of NumTotal vs Target  ----------------------------------------------------------------------#
st.subheader('Bivariate Analysis of NumTotal vs Target (Response)')

code = '''
# plot crosstab of NumTotal and response
pd.crosstab(df['NumTotal'],df['Response'],margins=True).head().style.background_gradient(cmap='Greys')
fig = plt.figure(figsize=(20, 6))
plt.title('Years being a customer')
ax = sns.barplot(x="NumTotal", y="Response", data=df)
'''
st.code(code, language='python')

pd.crosstab(df['NumTotal'],df['Response'],margins=True).head().style.background_gradient(cmap='Greys')
fig = plt.figure(figsize=(20, 6))
plt.title('Years being a customer')
ax = sns.barplot(x="NumTotal", y="Response", data=df)
st.pyplot(fig)

#--------------------------- Bivariate Analysis of Age vs Target  ----------------------------------------------------------------------#
st.subheader('Bivariate Analysis of Age vs Target (Response)')

code = '''
# plot crosstab of Age and response
pd.crosstab(df['Age'],df['Response'],margins=True).style.background_gradient(cmap='Greys')
fig = plt.figure(figsize=(20, 6))
plt.title('Years being a customer')
ax = sns.barplot(x="Age", y="Response", data=df)
'''
st.code(code, language='python')

pd.crosstab(df['Age'],df['Response'],margins=True).style.background_gradient(cmap='Greys')
fig = plt.figure(figsize=(20, 6))
plt.title('Years being a customer')
ax = sns.barplot(x="Age", y="Response", data=df)
st.pyplot(fig)

#--------------------------- Bivariate Analysis of YearsACustomer vs Target  ----------------------------------------------------------------------#
st.subheader('Bivariate Analysis of YearsACustomer vs Target (Response)')

code = '''
# plot crosstab of YearsACustomer and response
pd.crosstab(df['YearsACustomer'],df['Response'],margins=True).style.background_gradient(cmap='Greys')
fig = plt.figure(figsize=(20, 6))
plt.title('Years being a customer')
ax = sns.barplot(x="YearsACustomer", y="Response", data=df)
'''
st.code(code, language='python')

pd.crosstab(df['YearsACustomer'],df['Response'],margins=True).style.background_gradient(cmap='Greys')
fig = plt.figure(figsize=(20, 6))
plt.title('Years being a customer')
ax = sns.barplot(x="YearsACustomer", y="Response", data=df)
st.pyplot(fig)

















#--------------------------- CRM/CRF STATS  ----------------------------------------------------------------------#

st.header('CRM/RFM Analysis')

st.markdown("""
CRM/RFM analysis is a customer segmentation and analysis technique commonly used in marketing and customer relationship management (CRM) to understand and categorize customers based on their purchasing behavior and engagement with a business. CRM stands for Customer Relationship Management, and RFM stands for Recency, Frequency, and Monetary Value. Let's break down each component of RFM analysis:
- Recency (R): This factor assesses how recently a customer has made a purchase or interacted with the business. Customers who have interacted more recently are often considered more engaged and active.
- Frequency (F): Frequency measures how often a customer makes purchases or interacts with the business. Customers who make frequent purchases are generally more loyal and valuable.
- Monetary Value (M): This factor quantifies the total amount of money a customer has spent with the business. Customers with higher monetary value contribute more to the business's revenue.
- CRM/RFM analysis involves assigning numerical values to these three dimensions for each customer based on their transaction history. These values are then used to segment customers into different groups, often using a combination of thresholds or clustering algorithms. The resulting segments help businesses tailor their marketing and engagement strategies to different customer groups.
""")
            

code = '''
# define rfm dataframe
df['Frequency'] = df["NumTotal"]
rfm = df.loc[:, ['ID', 'Recency', 'Frequency', 'MntTotal']]
rfm.columns = ["ID",'recency', 'frequency', 'monetary']
'''
st.code(code, language='python')

df['Frequency'] = df["NumTotal"]
rfm = df.loc[:, ['ID', 'Recency', 'Frequency', 'MntTotal']]
st.dataframe(rfm)


rfm.columns = ["ID",'recency', 'frequency', 'monetary']

code = '''
# define scores for rfm
def rfm_scores(dataframe):
    dataframe["recency_score"] = pd.qcut(dataframe["recency"], 5, labels=[5, 4, 3, 2, 1])
    dataframe["frequency_score"] = pd.qcut(dataframe["frequency"], 5, labels=[1, 2, 3, 4, 5])
    dataframe["monetary_score"] = pd.qcut(dataframe["monetary"], 5, labels=[1, 2, 3, 4, 5])
    dataframe["rfm_score"] = dataframe["recency_score"].astype(str) + dataframe["frequency_score"].astype(str)
    return dataframe
'''
st.code(code, language='python')

def rfm_scores(dataframe):
    dataframe["recency_score"] = pd.qcut(dataframe["recency"], 5, labels=[5, 4, 3, 2, 1])
    dataframe["frequency_score"] = pd.qcut(dataframe["frequency"], 5, labels=[1, 2, 3, 4, 5])
    dataframe["monetary_score"] = pd.qcut(dataframe["monetary"], 5, labels=[1, 2, 3, 4, 5])
    dataframe["rfm_score"] = dataframe["recency_score"].astype(str) + dataframe["frequency_score"].astype(str)
    return dataframe
st.dataframe(rfm_scores(rfm))


st.markdown("""
Each segment represents a different type of customer with distinct characteristics and requires a tailored approach for marketing and customer relationship management. Here's a brief overview of each segment:
- Top-tier: They have bought recently, make frequent purchases, and spend the most. They are highly engaged and valuable to your business.
- Loyal: While not as high-spending as Champions, Loyal Customers still make frequent purchases and are responsive to promotions. They contribute consistently to your revenue.
- Potential: These customers are relatively new but have shown promising behavior. They've spent a good amount and made more than one purchase, indicating they have the potential to become Loyal Customers.
- New: These customers have made a recent purchase, but they don't buy very often. They could be new to your brand and might require targeted efforts to encourage repeat purchases.
- Recent: Similar to New Customers, Theses have made recent purchases, but their spending is relatively low. They could be enticed to spend more through targeted promotions or incentives.
- Near-recent: These customers have above-average recency, frequency, and monetary values, but they haven't made a purchase very recently. They might need a gentle nudge to re-engage with your brand.
- Near-disengagement: these ustomers have below-average recency, frequency, and monetary values. They are at risk of becoming less engaged, and efforts should be made to prevent their disengagement.
- Risk: These customers have historically been high spenders and frequent purchasers, but it's been a while since their last purchase. Efforts should be focused on reactivating them and bringing them back.
- Important-disengaged: These customers have made significant purchases in the past and were frequent buyers, but they haven't returned in a long time. Losing them would be a significant loss to your business.
- Disengaged: Hibernating customers have low recency, frequency, and monetary values. They've made only occasional purchases, and it's been a while since their last interaction.

These segments allow you to tailor your marketing strategies to each group's unique characteristics and needs, with the ultimate goal of maximizing customer engagement, retention, and revenue. It's important to regularly analyze and update these segments as customer behavior evolves over time.
            """)


code = '''
# create seg_map
seg_map = {
    r'[1-2][1-2]': 'Disengaged',
    r'[1-2][3-4]': 'Risk',
    r'[1-2]5': 'Important_disengaged',
    r'3[1-2]': 'Near_disengagement',
    r'33': 'Near_recent',
    r'[3-4][4-5]': 'Loyal',
    r'41': 'Recent',
    r'51': 'New',
    r'[4-5][2-3]': 'Potential',
    r'5[4-5]': 'Top_tier'
}
rfm['segment'] = rfm['rfm_score'].replace(seg_map, regex=True)
rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])


# plot the segmented costumer groups
plt.figure(figsize=(20,15))
fig, ax = plt.subplots(figsize=(20,10))
ax.grid(ls="dotted",lw=0.5,color="white",zorder=1);
bar_ap = sns.countplot(data = rfm,x = 'segment');
total = len(rfm['segment'])
for patch in ax.patches:
    percentage = '{:.1f}%'.format(100 * patch.get_height()/total)
    x = patch.get_x() + patch.get_width() / 2 - 0.17
    y = patch.get_y() + patch.get_height() * 1.005
    ax.annotate(percentage, (x, y), size = 14)
    
bar_ap.set_xticklabels(bar_ap.get_xticklabels(), rotation=90);
plt.title('Number of Customers by Segments', size = 16);
plt.ylabel('Count', size = 16,color="black")
plt.xlabel("Segment",fontsize = 16,color="black")
plt.xticks(size = 10)
plt.yticks(size = 10)
st.pyplot(fig)


'''
st.code(code, language='python')

seg_map = {
    r'[1-2][1-2]': 'Disengaged',
    r'[1-2][3-4]': 'Risk',
    r'[1-2]5': 'Important_disengaged',
    r'3[1-2]': 'Near_disengagement',
    r'33': 'Near_recent',
    r'[3-4][4-5]': 'Loyal',
    r'41': 'Recent',
    r'51': 'New',
    r'[4-5][2-3]': 'Potential',
    r'5[4-5]': 'Top_tier'
}
rfm['segment'] = rfm['rfm_score'].replace(seg_map, regex=True)
rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])

plt.figure(figsize=(20,15))
fig, ax = plt.subplots(figsize=(20,10))

ax.grid(ls="dotted",lw=0.5,color="white",zorder=1);
bar_ap = sns.countplot(data = rfm,x = 'segment');
total = len(rfm['segment'])
for patch in ax.patches:
    percentage = '{:.1f}%'.format(100 * patch.get_height()/total)
    x = patch.get_x() + patch.get_width() / 2 - 0.17
    y = patch.get_y() + patch.get_height() * 1.005
    ax.annotate(percentage, (x, y), size = 14)
    
bar_ap.set_xticklabels(bar_ap.get_xticklabels(), rotation=90);
plt.title('Number of Customers by Segments', size = 16);
plt.ylabel('Count', size = 16,color="black")
plt.xlabel("Segment",fontsize = 16,color="black")
plt.xticks(size = 10)
plt.yticks(size = 10)
st.pyplot(fig)








st.header('Encoding Categorical Features')
#--------------------------- Preprocessing (Encoding)  ----------------------------------------------------------------------#
code = '''
# import labelEncoder
from sklearn.preprocessing import LabelEncoder

# Encode Education and Marital_Status - 0,1,2
encode = LabelEncoder()
for i in ['Education', 'Marital_Status']:
    df[i]=df[[i]].apply(encode.fit_transform)
'''
st.code(code, language='python')

from sklearn.preprocessing import LabelEncoder
encode = LabelEncoder()
for i in ['Education', 'Marital_Status']:
    df[i]=df[[i]].apply(encode.fit_transform)



st.header('Correlation Analysis Between features')
#--------------------------- Correlation Analysis  ----------------------------------------------------------------------#
code = '''
# remove reduced features and unvaluable features
col_del = ["ID","NumWebVisitsMonth", "NumWebPurchases","NumCatalogPurchases","NumStorePurchases","NumDealsPurchases" , "Kidhome", "Teenhome","MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds","Dt_Customer"]
df=df.drop(columns=col_del,axis=1)
'''
st.code(code, language='python')

col_del = ["ID","NumWebVisitsMonth", "NumWebPurchases","NumCatalogPurchases","NumStorePurchases","NumDealsPurchases" , "Kidhome", "Teenhome","MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds","Dt_Customer"]
df=df.drop(columns=col_del,axis=1)
st.dataframe(df)


code = '''
# plot heatmap of correlations
fig = plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True)
st.pyplot(fig)
'''
st.code(code, language='python')

fig = plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True)
st.pyplot(fig)



st.header('Non-PCA Clustering')
#--------------------------- CLUSTERS  ----------------------------------------------------------------------#

st.markdown("""
Clustering is a data analysis technique used in customer segmentation to group similar customers together based on certain attributes or features. The goal of clustering is to identify patterns and relationships within a dataset without using predefined categories. In the context of customer segmentation, clustering helps to divide a customer base into distinct groups that share similar characteristics, behaviors, or preferences. This allows businesses to better understand their customers and tailor marketing strategies to specific groups.
- Choosing a Clustering Algorithm: Select an appropriate clustering algorithm based on the nature of your data and your objectives. Common clustering algorithms include K-Means, Hierarchical Clustering, DBSCAN, and Gaussian Mixture Models.
- Setting Parameters: For algorithms like K-Means, you'll need to specify the number of clusters you want to create. This can be determined using domain knowledge, business goals, or techniques like the Elbow Method or Silhouette Score.       
- Marketing Strategy Implementation: Tailor marketing campaigns, product recommendations, and customer engagement strategies based on the insights gained from each cluster. Different clusters may require different approaches to effectively engage and retain customers.            
            """)

df_cluster = df.copy()

st.subheader('Clustering using KMeans')
######################## KMEANS

code = '''
# make a copy of the original dataframe for data safety
df_cluster = df.copy()


# import visualizers and cluster algo
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans

# scale the dataframe
cluster_data = df.values
std_scale = StandardScaler().fit(cluster_data)
cluster_data = std_scale.transform(cluster_data)

# pick the k value using inertia
inertia = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 10)
    kmeans.fit(cluster_data)
    inertia.append(kmeans.inertia_)
    
elbow = 2
fig = plt.figure()
plt.plot(range(1, 10), inertia, marker = '*', alpha=0.5)
plt.scatter(elbow, inertia[elbow-1], s=100, c='r', marker='*')
plt.xlabel('Clusters')
plt.ylabel('Inertia')
plt.annotate('Elbow Point' ,(elbow, inertia[elbow-1]), xytext=(elbow,inertia[elbow-1] + 3000))
'''
st.code(code, language='python')

from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
cluster_data = df.values
std_scale = StandardScaler().fit(cluster_data)
cluster_data = std_scale.transform(cluster_data)

inertia = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 10)
    kmeans.fit(cluster_data)
    inertia.append(kmeans.inertia_)
    
elbow = 2
fig = plt.figure()
plt.plot(range(1, 10), inertia, marker = '*', alpha=0.5)
plt.scatter(elbow, inertia[elbow-1], s=100, c='r', marker='*')
plt.xlabel('Clusters')
plt.ylabel('Inertia')
plt.annotate('Elbow Point' ,(elbow, inertia[elbow-1]), xytext=(elbow,inertia[elbow-1] + 3000))
st.pyplot(fig)


code = '''
# start the clustering model and visualizer
model = KMeans(init = 'k-means++')
k_lst = []

# perform K-means in 4 different clusters to get mean k, plot each
fig = plt.figure(figsize=(15,10))
plt.subplot(221)
visualizer = KElbowVisualizer(model, k=(2,15), metric='distortion')
visualizer.fit(cluster_data)        # Fit the data to the visualizer
visualizer.finalize()
k_lst.append(visualizer.elbow_value_)

plt.subplot(222)
visualizer = KElbowVisualizer(model, k=(2,15), metric='distortion')
visualizer.fit(cluster_data)        # Fit the data to the visualizer
visualizer.finalize()
k_lst.append(visualizer.elbow_value_)

plt.subplot(223)
visualizer = KElbowVisualizer(model, k=(2,15), metric='distortion')
visualizer.fit(cluster_data)        # Fit the data to the visualizer
visualizer.finalize()
k_lst.append(visualizer.elbow_value_)

plt.subplot(224)
visualizer = KElbowVisualizer(model, k=(2,15), metric='distortion')
visualizer.fit(cluster_data)        # Fit the data to the visualizer
visualizer.finalize()
k_lst.append(visualizer.elbow_value_)

# print final mean  K
print('Mean K: ', np.mean(k_lst))
'''
st.code(code, language='python')

# Instantiate the clustering model and visualizer
model = KMeans(init = 'k-means++')
k_lst = []

# perform K-means 4 times(different intial clusters)
fig = plt.figure(figsize=(15,10))
plt.subplot(221)
visualizer = KElbowVisualizer(model, k=(2,15), metric='distortion')
visualizer.fit(cluster_data)        # Fit the data to the visualizer
visualizer.finalize()
k_lst.append(visualizer.elbow_value_)

plt.subplot(222)
visualizer = KElbowVisualizer(model, k=(2,15), metric='distortion')
visualizer.fit(cluster_data)        # Fit the data to the visualizer
visualizer.finalize()
k_lst.append(visualizer.elbow_value_)

plt.subplot(223)
visualizer = KElbowVisualizer(model, k=(2,15), metric='distortion')
visualizer.fit(cluster_data)        # Fit the data to the visualizer
visualizer.finalize()
k_lst.append(visualizer.elbow_value_)

plt.subplot(224)
visualizer = KElbowVisualizer(model, k=(2,15), metric='distortion')
visualizer.fit(cluster_data)        # Fit the data to the visualizer
visualizer.finalize()
k_lst.append(visualizer.elbow_value_)

st.pyplot(fig)
print('Mean K: ', np.mean(k_lst))



code = '''
# Do both Calinski_harabasz and Silhouette Scoring Matrix
fig = plt.figure(figsize=(18,5))

plt.subplot(121)
visualizer = KElbowVisualizer(model, k=(2,15), metric='calinski_harabasz')
visualizer.fit(cluster_data)        # Fit the data to the visualizer
visualizer.finalize()

plt.subplot(122)
visualizer = KElbowVisualizer(model, k=(2,15), metric='silhouette')
visualizer.fit(cluster_data)        # Fit the data to the visualizer
visualizer.finalize()
'''
st.code(code, language='python')




# Calinski_harabasz Scoring Matrix
fig = plt.figure(figsize=(18,5))

plt.subplot(121)
visualizer = KElbowVisualizer(model, k=(2,15), metric='calinski_harabasz')
visualizer.fit(cluster_data)        # Fit the data to the visualizer
visualizer.finalize()

# Silhouette Scoring Matrix
plt.subplot(122)
visualizer = KElbowVisualizer(model, k=(2,15), metric='silhouette')
visualizer.fit(cluster_data)        # Fit the data to the visualizer
visualizer.finalize()
st.pyplot(fig)




code = '''
# fit the final K-Means Model
kmeans = KMeans(n_clusters=2, init = 'k-means++').fit(cluster_data)
pred = kmeans.predict(cluster_data)
df_cluster['Cluster'] = pred + 1
'''
st.code(code, language='python')

# Building & Fitting K-Means Models
kmeans = KMeans(n_clusters=2, init = 'k-means++').fit(cluster_data)
pred = kmeans.predict(cluster_data)
df_cluster['Cluster'] = pred + 1
st.dataframe(df_cluster)


code = '''
# plot final clusters (Income & MntTotal)
fig = plt.figure()
sns.scatterplot(x='Income',y='MntTotal',hue='Cluster',data=df_cluster)
'''
st.code(code, language='python')

fig = plt.figure()
sns.scatterplot(x='Income',y='MntTotal',hue='Cluster',data=df_cluster)
st.pyplot(fig)


code = '''
# plot the clusters for each feature in the dataset
for i in df_cluster:
    g = sns.FacetGrid(df_cluster, col = "Cluster", hue = "Cluster", sharey=False, sharex=False)
    g.map(sns.histplot,i) 
    
    g.set_xticklabels(rotation=30)
    g.set_yticklabels()
    g.fig.set_figheight(5)
    g.fig.set_figwidth(20)
'''
st.code(code, language='python')

"""
for i in df_cluster:
    g = sns.FacetGrid(df_cluster, col = "Cluster", hue = "Cluster", sharey=False, sharex=False)
    g.map(sns.histplot,i) 
    
    g.set_xticklabels(rotation=30)
    g.set_yticklabels()
    g.fig.set_figheight(5)
    g.fig.set_figwidth(20)
    st.pyplot(g)
"""




st.subheader("Cluster using Gaussian Mixture Model")
###############  Gaussian Mixture Model


code = '''
# import the algo
from sklearn.mixture import GaussianMixture
log_like_lst = []
all_cluster = 15

# fit and plot the elbow point
for k in range(2, all_cluster):
    gmm = GaussianMixture(n_components = k, random_state = 100).fit(cluster_data)
    log_like = gmm.bic(cluster_data)
    log_like_lst.append(log_like)

elbow = 8
fig = plt.figure(figsize=(18,5))
plt.plot(range(2, all_cluster), log_like_lst, alpha=0.5)
plt.scatter(elbow, log_like_lst[elbow-2], s=100, c='r', marker='*')
plt.ylabel('BIC')
plt.xlabel('K')
plt.annotate('Elbow Point' ,(elbow, log_like_lst[elbow-1]), xytext=(elbow - 0.5,log_like_lst[elbow-2] + 3000))
'''
st.code(code, language='python')


from sklearn.mixture import GaussianMixture
log_like_lst = []
all_cluster = 15

for k in range(2, all_cluster):
    gmm = GaussianMixture(n_components = k, random_state = 100).fit(cluster_data)
    log_like = gmm.bic(cluster_data)
    log_like_lst.append(log_like)

elbow = 8
fig = plt.figure(figsize=(18,5))
plt.plot(range(2, all_cluster), log_like_lst, alpha=0.5)
plt.scatter(elbow, log_like_lst[elbow-2], s=100, c='r', marker='*')
plt.ylabel('BIC')
plt.xlabel('K')
plt.annotate('Elbow Point' ,(elbow, log_like_lst[elbow-1]), xytext=(elbow - 0.5,log_like_lst[elbow-2] + 3000))
st.pyplot(fig)



code = '''
# final model and fit
gmm = GaussianMixture(n_components = 8, random_state = 100).fit(cluster_data)
labels = gmm.predict(cluster_data)
df_cluster['Cluster_GMM'] = labels + 1
'''
st.code(code, language='python')

gmm = GaussianMixture(n_components = 8, random_state = 100).fit(cluster_data)
labels = gmm.predict(cluster_data)
df_cluster['Cluster_GMM'] = labels + 1
st.dataframe(df_cluster)

code = '''
# plot Income and Age Clusters
fig = plt.figure()
sns.scatterplot(x='Income',y='Age',hue='Cluster_GMM',data=df_cluster)
'''
st.code(code, language='python')

fig = plt.figure()
sns.scatterplot(x='Income',y='Age',hue='Cluster_GMM',data=df_cluster)
st.pyplot(fig)



code = '''
# plot clusters in every feature of dataset
for i in df_cluster:
    if i == 'Cluster':
        continue
    g = sns.FacetGrid(df_cluster, col = "Cluster_GMM", hue = "Cluster_GMM", sharey=False, sharex=False)
    g.map(sns.histplot,i) 
    g.set_xticklabels(rotation=30)
    g.set_yticklabels()
    g.fig.set_figheight(5)
    g.fig.set_figwidth(20)
'''
st.code(code, language='python')

"""
for i in df_cluster:
    if i == 'Cluster':
        continue
    g = sns.FacetGrid(df_cluster, col = "Cluster_GMM", hue = "Cluster_GMM", sharey=False, sharex=False)
    g.map(sns.histplot,i) 
    g.set_xticklabels(rotation=30)
    g.set_yticklabels()
    g.fig.set_figheight(5)
    g.fig.set_figwidth(20)
    st.pyplot(g)
"""


st.header('Clustering after PCA')

st.markdown("""
Whether to perform clustering before or after applying Principal Component Analysis (PCA) depends on the specific goals of your analysis and the characteristics of your data. Both approaches have their advantages and considerations. Let's explore both scenarios:

- Clustering Before PCA:
*Advantages:* Clustering on the original features allows you to retain the interpretability of the original variables. You can directly analyze the clusters in terms of the original features.
It might be easier to explain and communicate the results of clustering when the features have a clear and meaningful interpretation.
*Considerations:* Clustering on high-dimensional data can be computationally intensive and might lead to clusters that are highly influenced by noise or irrelevant features.
High-dimensional data might suffer from the "curse of dimensionality," where the distances between data points become less informative as the number of dimensions increases.
*Use Case:* If you have a small number of features and believe that they are highly relevant for clustering, you might choose to cluster before PCA.


- Clustering After PCA:

*Advantages:* PCA can reduce the dimensionality of the data by capturing the most important information while reducing noise and redundancy. This can improve the effectiveness of clustering.
Clustering in the reduced-dimensional space can help mitigate the computational challenges associated with high-dimensional data.
PCA might help reveal underlying patterns that are not apparent in the original feature space.
*Considerations:* Clustering in the reduced space might make the interpretation of the clusters less intuitive, as they are defined by combinations of the original features.
It might be harder to explain the results of clustering to non-technical stakeholders if the features are abstract combinations from PCA.
*Use Case:* If you have a large number of features, some of which might be noisy or redundant, using PCA to reduce dimensionality before clustering can lead to more effective and stable clusters.
In practice, you might want to experiment with both approaches and compare the results. You could evaluate the stability and interpretability of clusters, as well as their usefulness for your specific business goals. If you choose to apply PCA before clustering, you can also consider how many principal components to retain based on the cumulative explained variance or other criteria.

Ultimately, the decision of whether to perform clustering before or after PCA depends on the trade-offs between interpretability, computational efficiency, and the characteristics of your data.     
            """)
#--------------------------- CLUSTER WITH PCA  ----------------------------------------------------------------------#


code = '''
# make a copy of the orifinal dataframe for data safety
ds = df.copy()

# scale the dataframe
scaler = StandardScaler()
scaler.fit(ds)
scaled_ds = pd.DataFrame(scaler.transform(ds),columns= ds.columns )
'''
st.code(code, language='python')

ds = df.copy()
scaler = StandardScaler()
scaler.fit(ds)
scaled_ds = pd.DataFrame(scaler.transform(ds),columns= ds.columns )
st.dataframe(scaled_ds)


code = '''
# do PCA to 3 dimensions and plot in 3D
pca = PCA(n_components=3)
pca.fit(scaled_ds)
PCA_ds = pd.DataFrame(pca.transform(scaled_ds), columns=(["col1","col2", "col3"]))
PCA_ds.describe().T

# 3D Projection of data after reduction
x =PCA_ds["col1"]
y =PCA_ds["col2"]
z =PCA_ds["col3"]
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(x,y,z, c="maroon", marker="o" )
ax.set_title("3D Projection of data after reduction")
'''
st.code(code, language='python')

#Initiating PCA to reduce dimentions aka features to 3
pca = PCA(n_components=3)
pca.fit(scaled_ds)
PCA_ds = pd.DataFrame(pca.transform(scaled_ds), columns=(["col1","col2", "col3"]))
PCA_ds.describe().T

x =PCA_ds["col1"]
y =PCA_ds["col2"]
z =PCA_ds["col3"]

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(x,y,z, c="maroon", marker="o" )
ax.set_title("3D Projection of data after reduction")
st.pyplot(fig)


code = '''
# Elbow Method to determinek
fig = plt.figure(figsize=(10,8))
Elbow_M = KElbowVisualizer(KMeans(), k=10)
Elbow_M.fit(PCA_ds)
'''
st.code(code, language='python')


# Elbow Method to determine k
fig = plt.figure(figsize=(10,8))
Elbow_M = KElbowVisualizer(KMeans(), k=10)
Elbow_M.fit(PCA_ds)
#Elbow_M.show()
st.pyplot(fig)




code = '''
##Agglomerative Clustering model 
AC = AgglomerativeClustering(n_clusters=4)
yhat_AC = AC.fit_predict(PCA_ds)
PCA_ds["Clusters"] = yhat_AC
#Adding the Clusters feature to the orignal dataframe.
ds["Clusters"]= yhat_AC
cmap = colors.ListedColormap(["#682F2F", "#9E726F", "#D6B2B1", "#B9C0C9", "#9F8A78", "#F3AB60"])

fig = plt.figure(figsize=(10,8))
ax = plt.subplot(111, projection='3d', label="bla")
ax.scatter(x, y, z, s=40, c=PCA_ds["Clusters"], marker='o', cmap = cmap)
ax.set_title("The Plot Of The Clusters")
'''
st.code(code, language='python')

#Agglomerative Clustering model 
AC = AgglomerativeClustering(n_clusters=4)
yhat_AC = AC.fit_predict(PCA_ds)
PCA_ds["Clusters"] = yhat_AC
#Adding the Clusters feature to the orignal dataframe.
ds["Clusters"]= yhat_AC
cmap = colors.ListedColormap(["#682F2F", "#9E726F", "#D6B2B1", "#B9C0C9", "#9F8A78", "#F3AB60"])

fig = plt.figure(figsize=(10,8))
ax = plt.subplot(111, projection='3d', label="bla")
ax.scatter(x, y, z, s=40, c=PCA_ds["Clusters"], marker='o', cmap = cmap)
ax.set_title("The Plot Of The Clusters")
st.pyplot(fig)


code = '''
# countplot of clusters
fig = plt.figure(figsize=(10,8))
pal = ["#682F2F","#B9C0C9", "#9F8A78","#F3AB60"]
pl = sns.countplot(x=ds["Clusters"], palette= pal)
pl.set_title("Distribution Of The Clusters")
st.pyplot(fig)

fig = plt.figure(figsize=(10,8))
pl = sns.scatterplot(data = ds,x=ds["MntTotal"], y=ds["Income"],hue=ds["Clusters"], palette= pal)
pl.set_title("Cluster's Profile Based On Income And Spending")
plt.legend()
'''
st.code(code, language='python')

fig = plt.figure(figsize=(10,8))
pal = ["#682F2F","#B9C0C9", "#9F8A78","#F3AB60"]
pl = sns.countplot(x=ds["Clusters"], palette= pal)
pl.set_title("Distribution Of The Clusters")
st.pyplot(fig)

fig = plt.figure(figsize=(10,8))
pl = sns.scatterplot(data = ds,x=ds["MntTotal"], y=ds["Income"],hue=ds["Clusters"], palette= pal)
pl.set_title("Cluster's Profile Based On Income And Spending")
plt.legend()
st.pyplot(fig)


#--------------------------- POFILING  ----------------------------------------------------------------------#

"""Personal = [ "YearsACustomer", "Age", "Kids", "Education","Marital_Status"]

for i in Personal:
    fig = plt.figure()
    sns.jointplot(x=ds[i], y=ds["Response"], hue =ds["Clusters"], kind="kde", palette=pal)
    st.pyplot(fig)"""

#--------------------------- PREDICTION MODEL  ----------------------------------------------------------------------#

st.dataframe(ds)
df.to_csv("original.csv", sep='\t')
