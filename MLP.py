import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from joblib import dump
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder

#Read in data formatted by the MultiLabelBinarizer (MLB) CSV
team_comps = pd.read_csv("filteredData/MultiLabelBinarizer.csv")

#Splits data into features and target.
#Features are the inputs to the predictors, while targets are the expected outcome for each feature.
#Currently the features are just team composition, so I drop the match_id and targets from the dataframe.
Comp_Features = team_comps.drop(['match_id','match_win', 'map_win'], axis = 1)
Comp_TargetMatch = team_comps['match_win']
Comp_TargetMap = team_comps['map_win']

#Splits the features and targets into a training set and testing set.
Features_train, Features_test, Target_train, Target_test = train_test_split(Comp_Features, Comp_TargetMap, test_size = 0.20 )
'''
#This splits dataframe into features and targets with maps included in the MLB
comp_and_map = pd.read_csv("filteredData/MultiLabelBinarizer2.csv")
Features = comp_and_map.drop(['map_result'], axis = 1)
TargetMap = comp_and_map['map_result']

Features_train2, Features_test2, Target_train2, Target_test2 = train_test_split(Features, TargetMap, test_size = 0.20 )
'''

#Creates a pipeline that encodes the data and then trains the associated classifer with the data.
pipe = Pipeline([
    ('encoder', OneHotEncoder(handle_unknown='ignore')),
    ('svc', svm.SVC(kernel = 'linear', C=1, probability=True))
    ])

pipe2 = Pipeline([
    ('encoder', OneHotEncoder(handle_unknown='ignore')),
    ("knn", KNeighborsClassifier(n_neighbors= 5))
])

pipe3 = Pipeline([
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse = False)),
    ("nb", GaussianNB())
])

pipe4 = Pipeline([
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse = False)),
    ("lda", LinearDiscriminantAnalysis())
])

################################################
print("Fitting SVC... ")
#Fits each pipeline to the input features.
svc = pipe.fit(Features_train, Target_train)

print("\nFitting KNN... ")
knn = pipe2.fit(Features_train, Target_train)

print("\nFitting Naive-Bayes... ")
nb = pipe3.fit(Features_train, Target_train)

print("\nFitting LDA... ")
lda = pipe4.fit(Features_train, Target_train)

################################################
#Calculates the accuracy accuracy of each classifier.

print("\nSVC accuracy: ")
print(svc.score(Features_test, Target_test))

print("\nKNN accuracy: ")
print(knn.score(Features_test, Target_test))

print("\nNaive-Bayes accuracy: ")
print(nb.score(Features_test, Target_test))

print("\nLDA accuracy: ")
print(lda.score(Features_test, Target_test))

################################################
#Saves each classifier to the models folder. 
dump(svc, "models/svc.joblib")
dump(knn, "models/knn.joblib")
dump(nb, "models/nb.joblib")
dump(lda, "models/lda.joblib")

print("\nModels saved.")
