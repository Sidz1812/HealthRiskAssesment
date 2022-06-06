#!/usr/bin/env python
# coding: utf-8


# load the required library for data preprocessing
import numpy as np
import pandas as pd
import warnings
from sklearn.cluster import KMeans


# read the data from csv file as dataframe
data_2018 = pd.read_csv('~/working_project/2018_Health_Evals.csv', low_memory=False)
data_2017 = pd.read_csv('~/working_project/2017_Health_Evals.csv', low_memory=False)
data_2016 = pd.read_csv('~/working_project/2012-2016_Health_Evals.csv', low_memory=False)

# combine all the dataframe as one dataframe for preprocessing
data_all = pd.concat([data_2018, data_2017, data_2016], sort=False)


#Feature Engineering

# select columns for analysis
data_1 = data_all.iloc[:, 26:80]
data_2 = data_all.iloc[:, 84:86]
data = pd.concat([data_1, data_2], axis=1, sort=False)

# delete columns that we don't need for model training
data.drop(columns=['PrimaryCareProvider', "COTININE", "A1c", "DiabetesStatusUsingA1c", 'SnoringHistory',
      "Mammography", "PapSmear", "ClinicalBreastExam", "ColorectalExam", "DiabetesDuringPregnancyHistory",
      'DaytimeFatigueHistory', 'KidneyHistory', 'StrokeHistory', 'HypertensionMedication', 'DiabetesMedication'], inplace=True)


# data cleaning step
# we excluded patients who didn't fast,
# and patients with negative age or negative LDL level, or negative metabolic syndrome risk


data.drop(data[data["Fasting"] == 'NO'].index, inplace=True)
data.drop(data[data["Fasting"].isnull()].index, inplace=True)
data.drop(data[data["Age"] < 0].index, inplace=True)
data.drop(data[data["METS_Risks"] == -1].index, inplace=True)
data.drop(data[data["LDL"] < 0].index, inplace=True)


# map the ordinal assessment results to numerical value
maps_smk = {'NEVER': 0, 'QUIT': 1, 'YES': 2, 'NO': 0}
data["Smoking"] = data["Smoking"].map(maps_smk)
data['SmokelessTobacco'] = data['SmokelessTobacco'].map(maps_smk)


maps_binary = {'YES': 1, 'NO': 0}
col_binary = ["Activity", "Alcohol"]
col_history = ['HypertensionHistory',
       'DiabetesHistory', 'CoronaryArteryHistory', 'HighCholesterolHistory',
       'AsthmaHistory', 'HeartFailureHistory', 'COPDHistory',
       'HeartDiseaseHistory',]

for cols in col_binary:
    data[cols] = data[cols].map(maps_binary)


for col in col_history:
    data[col]=np.where(data[col].isnull(), "NO", data[col])
    data[col]=data[col].map(maps_binary)
#     print(data[col].value_counts())
#     print(data[col].isnull().sum())

# drop missing value
data.dropna(inplace=True)


# drop patients who didn't fast
col_nofast=["HDLAssessment", "LDLAssessment", "DiabetesStatusUsingGlucose", "TGSAssessment"]
for col in col_nofast:
    data.drop(data[data[col]=='NOT FASTING'].index, inplace=True)

#"BPAssessment"
data["BPAssessment"] = data["BPAssessment"].map({'CONTROLLED':3, 'NORMAL':0, 'HYPERTENSION STG1':2, 'HYPERTENSION STG2':4, 'ELEVATED':1, 'HYPERTENSIVE CRISIS':5})
#"HDLAssessment"
data["HDLAssessment"] = data["HDLAssessment"].map({'VERY GOOD':0, 'ACCEPTABLE':1, 'LOW':2})
#"LDLAssessment"
data["LDLAssessment"] = data["LDLAssessment"].map({'OPTIMAL':0, 'GOOD':1, 'BORDERLINE HIGH':2, 'CONTROLLED':3, 'HIGH':4})
#"TGSAssessment"
data["TGSAssessment"] = data["TGSAssessment"].map({'NORMAL':0, 'BORDERLINE HIGH':1, 'HIGH':2, 'VERY HIGH':3})
#"DMAssessment","DiabetesStatusUsingGlucose"
maps_dm = {'NORMAL':0, 'PRE-DIABETES':2, 'DIABETES':3, 'MANAGED':1}
data["DMAssessment"]=data["DMAssessment"].map(maps_dm)
data["DiabetesStatusUsingGlucose"]=data["DiabetesStatusUsingGlucose"].map(maps_dm)
#BMI
maps_bmi = {'EXTREME OBESITY':4, 'NORMAL':0, 'OBESITY':3, 'OVERWEIGHT':2, 'UNDERWEIGHT':1}
data["BMIAssessment"]=data["BMIAssessment"].map(maps_bmi)
#"AbdominalCircumferenceStatus"
maps_abdom = {'HIGH RISK':1, 'NORMAL':0}
data["AbdominalCircumferenceStatus"] = data["AbdominalCircumferenceStatus"].map(maps_abdom)
#"Gender"
maps_g = {'F': 1, 'M': 0}
data["Gender"]=data["Gender"].map(maps_g)


# create a flag of high risk
data.loc[(data['EmergencyReferral'] == 'YES') | (data['HighRisk'] == 'YES'), 'Flag'] = 1
data.loc[(data['EmergencyReferral'] == 'NO') & (data['HighRisk'] == 'NO'), 'Flag'] = 0
# print(data['Flag'].value_counts())
# print(data['Flag'].isnull().sum())

#data.head(1)
# save the preprocessed data to csv file
#data.to_csv('~/working_project/production/cleaned.csv', index=False)


# define the target variable we would like the machine learning model to predict
y = data["Flag"]

# use the top 16 important features as predictors
cols=['SBP', 'DBP', 'BPAssessment', 'GLU', 'BMI', 'TGS', 'Age', 'TCHOL', 'AST', 'LDL', 'ALT', 'WEIGHT',
      'DMAssessment', 'AbdominalCir', 'METS_Risks', 'DiabetesStatusUsingGlucose']
X = data[cols]

# scale the features into range between 0 and 1
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_rescale = mms.fit_transform(X)


# store the parameters for scaling the new patients data
# if we retrain the model, update the variables of scaler_min and scaler_max in predict_model.py with the new mms.data_min_, mms.data_max_
# print(mms.data_min_)
# print(mms.data_max_)
# scaler_min = mms.data_min_
# scaler_max = mms.data_max_

# split the dataset into traning set and testing set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_rescale, y, test_size=0.3, random_state=0)


# set running_kmeans = True, if we need to retrain the model, so that we can update the nearest neighbors data for imputation
# for any missing values in new patients data:
# first, use k means clustering algorithm to group patients into 50 groups based on patients' similar characteristics
# second, calculate the center for each groups
# third, save the center data for prediction
# during the prediction process, if there any missing values in any features,
# use the manhattan distance to find the nearest neighbor from one of 50 groups,
# impute the missing values with the patient's nearest neighbors data for prediction
running_kmeans = False
if running_kmeans:
    kmeans = KMeans(n_clusters=50)
    kmeans.fit(X_test)
    centers = np.round(kmeans.cluster_centers_, decimals = 4)
    centers_df = pd.DataFrame(centers, columns=cols)
    centers_df.to_csv("centers.csv", index = False)



# apply the undersampling method to the training data set
import imblearn
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=0)
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)


# define a function to return performance metrics
def ConfusionMatrix_Report(y_test, y_predicted):
    confusion = confusion_matrix(y_test, y_predicted)
    ACC=accuracy_score(y_test, y_predicted)
    Precision=precision_score(y_test, y_predicted)
    Recall=recall_score(y_test, y_predicted)
    F1=f1_score(y_test, y_predicted)
    print('---Confusion Matrix---\n', confusion)
    print('\n   Accuracy: {:.2f}'.format(ACC))
    print('\n   Precision: {:.2f}'.format(Precision))
    print('\n   Recall: {:.2f}'.format(Recall))
    print('\n   F1: {:.2f}'.format(F1))
    print('---Classification Report---')
    print('\n   \n', 
    classification_report(y_test, y_predicted, target_names = ['not 1', '1']))


# define a function to return AUC
def ROC_decison_function(y_test, y_score):
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    print('AUC for GBT is', '% 0.2f' %roc_auc)
#     plt.figure()
#     plt.xlim([-0.01, 1.00])
#     plt.ylim([-0.01, 1.01])
#     plt.plot(fpr, tpr, lw=3, label='(AUC = {:0.2f})'.format(roc_auc))
#     plt.xlabel('False Positive Rate', fontsize=16)
#     plt.ylabel('True Positive Rate', fontsize=16)
#     plt.title('classifier ROC curve )', fontsize=16)
#     plt.legend(loc='riht', fontsize=13)
#     plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
#     plt.axes().set_aspect('equal')
#     plt.show()


# import the sklearn libraries for modelling
# call the gradient boosting classifier to fit the training data set
# store the model parameters in clf
# call clf to make prediction on testing data set to evaluate the model performance
# return the performance metrics
# model hyperparameters (learning rate, maximum depth, number of estimators) obtained from hyperparameters tuning process

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
clf = GradientBoostingClassifier(learning_rate=0.2, max_depth=3, n_estimators=60).fit(X_resampled, y_resampled)
clf_predicted = clf.predict(X_test)
y_score = clf.predict_proba(X_test)[:, 1]
ConfusionMatrix_Report(y_test=y_test,y_predicted=clf_predicted)
ROC_decison_function(y_test, y_score)


# import pickle library
import pickle

# save the pre-trained model into pickle format for production
with open('gbt.pickle','wb') as fw:
    pickle.dump(clf,fw)





# sample one record for testing the prediction
test_col = ['HealthEvalID','Fasting','SBP', 'DBP', 'BPAssessment', 'GLU', 'BMI', 'TGS', 'Age', 'TCHOL', 'AST', 'LDL', 'ALT', 'WEIGHT',
      'DMAssessment', 'AbdominalCir', 'METS_Risks', 'DiabetesStatusUsingGlucose','EmergencyReferral','HighRisk']
test_sample = data_all.sample(1)[test_col]
test_sample.to_json("test_data.json", orient='records')
