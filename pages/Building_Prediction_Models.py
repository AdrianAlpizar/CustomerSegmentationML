import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
import streamlit as st
import pandas as pd
import numpy as np


#Importing the Libraries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.decomposition import PCA  
import matplotlib.pyplot as plt, numpy as np 
from sklearn import metrics

#st.title('Building Prediction Models for Marketing Campaing Response')
original_df = pd.read_csv('original.csv', sep='\t', index_col="Unnamed: 0")


code = '''
# import the dataframe from previous notebook, clusters, feature engineering completed
original_df = pd.read_csv('original.csv', sep='\t', index_col="Unnamed: 0")
'''
st.code(code, language='python')
st.dataframe(original_df)


code = '''
# Use Random Forest Algo to find important features
clf = RandomForestClassifier()
clf.fit(original_df.drop('Response', axis=1), original_df['Response'])
importance = clf.feature_importances_
importance = pd.DataFrame(importance, index=original_df.drop('Response', axis=1).columns, columns=["Importance"]).sort_values(by='Importance', ascending=True)
st.dataframe(importance)
fig = plt.figure(figsize=(20,len(importance)/2))
ax = sns.barplot(x=importance['Importance'], y=importance['Importance'].index, data=importance, orient = 'h')
'''
st.code(code, language='python')

clf = RandomForestClassifier()
clf.fit(original_df.drop('Response', axis=1), original_df['Response'])
importance = clf.feature_importances_
importance = pd.DataFrame(importance, index=original_df.drop('Response', axis=1).columns, columns=["Importance"]).sort_values(by='Importance', ascending=True)
st.dataframe(importance)
fig = plt.figure(figsize=(20,len(importance)/2))
ax = sns.barplot(x=importance['Importance'], y=importance['Importance'].index, data=importance, orient = 'h')
st.pyplot(fig)




code = '''
# scale X, perform PCA and plot variance
std_scale = preprocessing.StandardScaler().fit(original_df.drop('Response', axis=1))
X = std_scale.transform(original_df.drop('Response', axis=1))

# perform PCA
pca1 = PCA(0.90, whiten=True) # Keep 90% information
fit1 = pca1.fit(X)

# plot variance of features
plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(25,7)) 
plt.xlabel('PCA Feature')
plt.ylabel('Variance')
plt.title('PCA for Whole Dataset')
plt.bar(range(0, fit1.explained_variance_ratio_.size), fit1.explained_variance_ratio_);
'''
st.code(code, language='python')


from sklearn.decomposition import PCA
# Calculating PCA for both datasets, and graphing the Variance for each feature, per dataset
std_scale = sklearn.preprocessing.StandardScaler().fit(original_df.drop('Response', axis=1))
X = std_scale.transform(original_df.drop('Response', axis=1))
pca1 = PCA(0.90, whiten=True) # Keep 90% information
fit1 = pca1.fit(X)

fig = plt.figure(figsize=(25,7)) 
plt.xlabel('PCA Feature')
plt.ylabel('Variance')
plt.title('PCA for Whole Dataset')
plt.bar(range(0, fit1.explained_variance_ratio_.size), fit1.explained_variance_ratio_);
st.pyplot(fig)



code = '''
# see dataframe after PCA
pca_data = pca1.transform(X)
pca_data = np.array(pca_data)
st.dataframe(pca_data)
st.write('PCA data shape: ', pca_data.shape)
'''
st.code(code, language='python')

pca_data = pca1.transform(X)
pca_data = np.array(pca_data)
st.dataframe(pca_data)
st.write('PCA data shape: ', pca_data.shape)


code = '''
# check imbalance on target variable
fig = plt.figure(figsize=(6,6))
original_df['Response'].value_counts().plot.pie(explode=[0.1,0.1], autopct='%1.1f%%', shadow=True, textprops={'fontsize':16}).set_title("Target distribution")
'''
st.code(code, language='python')

fig = plt.figure()
original_df['Response'].value_counts().plot.pie(explode=[0.1,0.1], autopct='%1.1f%%', shadow=True, textprops={'fontsize':10}).set_title("Target distribution")
st.pyplot(fig)


st.subheader("Training and testing Split ")

code = '''
dataset_num = 1
all_datasets = [original_df.drop('Response', axis=1).values, original_df.values, pca_data]
final_data = all_datasets[dataset_num]

# set X and y
X = final_data
y = original_df['Response'].values

# Target Variable has high imbalance use SMOTE to balance data
from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy=1, n_jobs=-1)
X, y = smote.fit_resample(X, y)

# split x and y to train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

'''
st.code(code, language='python')


perf_df_lst = [None, None, None] 
dataset_num = 1
all_datasets = [original_df.drop('Response', axis=1).values, original_df.values, pca_data]
final_data = all_datasets[dataset_num]


X = final_data
y = original_df['Response'].values

from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy=1, n_jobs=-1)
X, y = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


st.subheader("Testing Multiple Models using a Pipeline")

code = '''
# import libraries for testing different models
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# list all models to be used in pipeline
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

#set testing variables
num_folds = 10
seed = 7
scoring = 'accuracy'
results = []
names = []

# run pipeline
for name, model in models:
    kfold = KFold(n_splits=num_folds)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    print('{}: {} ({})'.format(name, cv_results.mean(), cv_results.std()))
'''
st.code(code, language='python')


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

num_folds = 10
seed = 7
scoring = 'accuracy'
results = []
names = []

for name, model in models:
    kfold = KFold(n_splits=num_folds)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    st.write('{}: {} ({})'.format(name, cv_results.mean(), cv_results.std()))

code = '''
# plot results
fig = plt.figure()
fig.suptitle( 'Algorithm Comparison' )
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)

'''
st.code(code, language='python')


fig = plt.figure()
fig.suptitle( 'Algorithm Comparison' )
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
st.pyplot(fig)


st.subheader("Testing AdaBoostClassifier")

code = '''
# set matthews_corrcoef metric
mcc_scorer = metrics.make_scorer(metrics.matthews_corrcoef)

# we will be testing AdaBoostClassifier 
model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2), n_estimators=160, random_state=1)  

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X_train, y_train, scoring=mcc_scorer, cv=cv, n_jobs=-1, error_score='raise')

# report performance
print('Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))
'''
st.code(code, language='python')


mcc_scorer = metrics.make_scorer(metrics.matthews_corrcoef)

#model = svm.SVC(kernel = 'rbf', C = 10, gamma = 0.01)
#model = LogisticRegression(solver='lbfgs', max_iter=1000)
model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2), n_estimators=160, random_state=1)  

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X_train, y_train, scoring=mcc_scorer, cv=cv, n_jobs=-1, error_score='raise')
# report performance
st.write('Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))



code = '''
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# plot Confusion Matrix
cm = sklearn.metrics.confusion_matrix(y_test,predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,  display_labels=clf.classes_)
st.pyplot(disp.plot())


'''
st.code(code, language='python')


model.fit(X_train, y_train)
predictions = model.predict(X_test)
cm = sklearn.metrics.confusion_matrix(y_test,predictions)
display = ConfusionMatrixDisplay(confusion_matrix=cm,  display_labels=clf.classes_)
fig, ax = plt.subplots()
display.plot(ax=ax)
st.pyplot(fig)


st.subheader("Testing a basic Deep Learning Model with Tensorflow")


code = '''

# import libraries
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# create badic model for accuracy testing
def create_baseline():

    # create model
    model = Sequential()
    model.add(Dense(14, input_shape=(14,), activation='relu'))
    model.add(Dense(7, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

'''
st.code(code, language='python')


import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# baseline model
def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(14, input_shape=(14,), activation='relu'))
    model.add(Dense(7, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

code = '''
# show accuracy results with kfolds and 5 epochs
estimator = KerasClassifier(model=create_baseline, epochs=5, batch_size=5, verbose=1)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, X_train, y_train, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
'''


estimator = KerasClassifier(model=create_baseline, epochs=10, batch_size=5, verbose=1)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, X_train, y_train, cv=kfold)
st.write("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

model = create_baseline()
history = model.fit(X_train, y_train,validation_data=(X_test, y_test), epochs=100, verbose=0)
train_mse = model.evaluate(X_train, y_train, verbose=0)
test_mse = model.evaluate(X_test, y_test, verbose=0)
#print(train_mse,test_mse)

code = '''
# plot validation loss vs loss
fig = plt.figure(figsize=(20,len(importance)/2))
plt.title('Loss / Mean Squared Error')
plt.plot(history.history['loss'], label='train')
print(history.history)
plt.plot(history.history['val_loss'], label='test')
plt.legend()
'''


fig = plt.figure(figsize=(20,len(importance)/2))
plt.title('Loss / Mean Squared Error')
plt.plot(history.history['loss'], label='train')
print(history.history)
plt.plot(history.history['val_loss'], label='test')
plt.legend()
st.pyplot(fig)
