import copy
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn import metrics
from sklearn.svm import LinearSVC
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from imblearn.pipeline import make_pipeline
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

#name the file
cyberBullyingFile  =  "sortedData.csv"

#load the file
data = pd.read_csv(cyberBullyingFile)
dataSetY = data.copy().ix[:,0]
dataSetX = data.copy().ix[:,1:data.shape[0]]

#Looks nice
print("Bullying classifications")

#Ive already sorted the data set around outward centrality so the halway
#point is the mean
def splitAtMedian(dSetX, dSetY) :
    introvertCopyX  = dSetX.copy()[0:int(dSetX.shape[0]/2)]
    extrovertCopyX  = dSetX.copy()[int(dSetX.shape[0]/2):int(dSetX.shape[0]-1)]

    introvertCopyY  = dSetY.copy()[0:int(dSetY.shape[0]/2)]
    extrovertCopyY  = dSetY.copy()[int(dSetY.shape[0]/2):int(dSetY.shape[0]-1)]

    return introvertCopyX , extrovertCopyX, introvertCopyY, extrovertCopyY

#run a 30 70 split with smote prepropressing
def crossValidate(X, Y) :
    sampler = SMOTE()
    Xa, Ya = sampler.fit_sample(X,Y)
    X_train, X_test, y_Train, y_test = train_test_split(Xa, Ya, test_size = 0.3, random_state = 1)

    return X_train, X_test, y_Train, y_test


#calculate statistical parity differences
def stasticalParity(dataI, labelsI, dataE, labelsE) :

    #Calculate the probability of being bullied vs not being bullied
    positiveProb = sum(1 for y in labelsI if y == 1) / len(dataI)
    elseProb = sum(1 for y in labelsE if y == 1) / len(dataE)

    #output the difference in statistical parity
    return elseProb - positiveProb

#calculate the difference needed to achieve equal opportunity
def calculateEqualOpportunity(confusionMatrixE, confusionMatrixI):

    #False negatives
    FNE = confusionMatrixE[0][1]
    FNI = confusionMatrixI[0][1]

    #True positives
    TPE = confusionMatrixE[0][0]
    TPI = confusionMatrixI[0][0]

    #True positive rates
    TPRE = TPE/(TPE+FNE)
    TPRI = TPI/(TPI+FNI)

    #return the difference in true positive rates
    return TPRE - TPRI


#splits introverts and extroverts at the median  for point of centrality
introvertSetX, extrovertSetX, introvertSetY, extrovertSetY = splitAtMedian(dataSetX, dataSetY)

#run cross validation
introXTrain, introXTest, introYTrain , introYTest = crossValidate(introvertSetX, introvertSetY)
extroXTrain, extroXTest, extroYTrain , extroYTest = crossValidate(extrovertSetX, extrovertSetY)

#train the classifier for introverts
rfI = RandomForestClassifier(n_estimators = 100, max_depth = 1, random_state = 1)
rfI.fit(introXTrain, introYTrain)

#calculate ROC for introverts
#y_pred_rfI = rfI.predict_proba(introXTest)[:, 1]
y_pred_rfI = rfI.predict(introXTest)
fprI, tprI, thresholdsI = metrics.roc_curve(introYTest, y_pred_rfI)

#train the classifier for Extroverts
rfE = RandomForestClassifier(n_estimators = 100, max_depth = 1 , random_state = 1)
rfE.fit(extroXTrain, extroYTrain)

#calculate ROC for Extroverts
#y_pred_rfE = rfE.predict_proba(extroXTest)[:, 1]
y_pred_rfE = rfE.predict(extroXTest)
fprE, tprE, thresholdsE = metrics.roc_curve(extroYTest, y_pred_rfE)

confusionMatrixExtro = metrics.confusion_matrix(extroYTest, y_pred_rfE)
confusionMatrixIntro = metrics.confusion_matrix(introYTest, y_pred_rfI)

print("Roc Auc Extrovert before preprocessing : ", roc_auc_score(extroYTest, y_pred_rfE))
print("Roc Auc Introverts before prepropressing : ", roc_auc_score(introYTest, y_pred_rfI))
print("statistical parity difference before preprocessing : ", stasticalParity(introvertSetX, introvertSetY, extrovertSetX, extrovertSetY))
print("Equal opportunity difference before preprocessing : ", calculateEqualOpportunity(confusionMatrixExtro, confusionMatrixIntro))

#random forest ROC graph for introverts
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fprI, tprI, label='ROC')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve Introverts')
plt.legend(loc='best')
plt.show()

#random forest ROC graph for Extroverts
plt.figure(2)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fprE, tprE, label='ROC')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve Extroverts')
plt.legend(loc='best')
plt.show()
