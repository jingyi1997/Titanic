import pandas as pd 
import numpy as np 
import os
import pickle
import operator
import matplotlib.pyplot as plt
import re
import sklearn.preprocessing as preprocessing
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve, RandomizedSearchCV, train_test_split
from sklearn import metrics
from scipy.stats import randint as sp_randint
from time import time
from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor, GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression
def pred_age(full_data):

    classers = ['Fare','Parch','Pclass','SibSp','TitleCat','CabinCat','female','male', 'EmbarkedCat', 'FamilySize', 'NameLength', 'FamilyId']
    age_et = ExtraTreesRegressor(n_estimators=200)
    print(full_data[full_data.Age.isnull()])
    X_train = full_data.loc[full_data.Age.notnull(),classers]
    
    Y_train = full_data.loc[full_data.Age.notnull(),['Age']]
    
    X_test = full_data.loc[full_data.Age.isnull(),classers]
    
    age_et.fit(X_train,np.ravel(Y_train))
    age_preds = age_et.predict(X_test)
    full_data.loc[full_data.Age.isnull(),['Age']] = age_preds


def get_title(name):
    # Use a regular expression to search for a title.  Titles always consist of capital and lowercase letters, and end with a period.
    title_search = re.search(' ([A-Za-z]+)\.', name)
    
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

def girl(aa):
    if (aa.Age!= 0)&(aa.Title=='Miss')&(aa.Age<=14):  
        return 'Girl'  
    elif (aa.Age == 0)&(aa.Title=='Miss')&(aa.Parch!=0):  
        return 'Girl'  
    else:  
        return aa.Title  
def encode_age(aa):
    if aa.Age > 0 and aa.Age < 14:
        return 0
    if aa.Age >= 14 and aa.Age < 30:
        return 1
    if aa.Age >= 30 and aa.Age < 45:
        return 2
    if aa.Age >= 45 and aa.Age < 60:
        return 3
    return 4

# generate new features 'title' and 'NameLength'
def convert_name(df):
    df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    
    df['Title'] = df['Title'].replace(['Capt', 'Col', 'Don', 'Dr','Jonkheer','Major', 'Rev', 'Sir'], 'Male_Rare')
    df['Title'] = df['Title'].replace(['Countess', 'Lady','Mlle', 'Mme', 'Ms', 'Dona'], 'Female_Rare')
    
    df['Title'] = df.apply(girl, axis = 1)
    
    

# fill missing ages according to title
def set_missing_ages_title(df):
    Tit=['Mr','Miss','Mrs','Master','Girl','Male_Rare','Female_Rare']  
    for i in Tit:   
        df.loc[(df.Age == 0)&(df.Title == i),'Age'] = df.loc[df.Title == i,'Age'].median()  
    
    return df


# filling missing ages according to  Pclass, Parch and SibSp
def set_missing_ages_pps(df):
    index_NaN_age = list(df["Age"][df["Age"].isnull()].index)
    print(index_NaN_age)
    for i in index_NaN_age :
        age_med = df["Age"].median()
        age_pred = df["Age"][((df['SibSp'] == df.iloc[i]["SibSp"]) & (df['Parch'] == df.iloc[i]["Parch"]) & (df['Pclass'] == df.iloc[i]["Pclass"]))].median()
        if not np.isnan(age_pred) :
            df.loc[i,'Age'] = age_pred
        else :
            df.loc[i,'Age'] = age_med   
        if( i == 413):
            print('age_med', age_med)
            print('age_pred', age_pred) 
    

#convert Cabin, Embarked, title to type
def convert_to_type(full_data):
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 7, "Dona":10, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 7, "Capt": 7, "Ms": 2}
    full_data["TitleCat"] = full_data.loc[:,'Title'].map(title_mapping)
    full_data['CabinCat'] = pd.Categorical.from_array(full_data.Cabin.fillna('0').apply(lambda x: x[0])).codes
    full_data.Cabin.fillna('0', inplace=True)
    full_data['EmbarkedCat'] = pd.Categorical.from_array(full_data.Embarked).codes
    full_data.drop(['Ticket'], axis=1, inplace=True)
    #print(df.head(10))

#generate new feature 'Family size'
def get_family_size(df):
    df['FamilySize'] = df["SibSp"] + df["Parch"] 
    df['NameLength'] = df.Name.apply(lambda x: len(x))
    # df['Single'] = df['FamilySize'].map(lambda s: 1 if s == 1 else 0)
    # df['SmallF'] = df['FamilySize'].map(lambda s: 1 if  s == 2  else 0)
    # df['MedF'] = df['FamilySize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
    # df['LargeF'] = df['FamilySize'].map(lambda s: 1 if s >= 5 else 0)
    



family_id_mapping = {}

# generate new feature 'Family Id'
def get_family_id(row):
    global family_id_mapping
    last_name = row["Name"].split(",")[0]
    family_id = "{0}{1}".format(last_name, row["FamilySize"])    
    if family_id not in family_id_mapping:
        if len(family_id_mapping) == 0:
            current_id = 1
        else:
            current_id = max(family_id_mapping.values()) + 1 
        family_id_mapping[family_id] = current_id
    return family_id_mapping[family_id]

def generate_family_id(df):
    family_ids = df.apply(get_family_id, axis=1)
    family_ids[df["FamilySize"] < 3] = -1
    df["FamilyId"] = family_ids
    




#### Person Label

def get_person(passenger):
    age, sex = passenger
    #print(age)
    child_age = 14
    if (age < child_age):
        return 'child'
    elif (sex == 'female'):
        return 'female_adult'
    else:
        return 'male_adult'


# generate new features 'person'
def generate_person_label(df):
    df['person'] = df[['Age', 'Sex']].apply(get_person, axis=1)
   






def surviving_first_class_male(passenger):
    Pclass, Sex, Survived = passenger
    if (Pclass == 1 and Sex == 'male' and Survived == 1.0):
        return 1.0
    else:
        return 0.0
def gen_first_class_male(df):
    df['surviving_first_class_male'] = df[['Pclass', 'Sex', 'Survived']].apply(surviving_first_class_male, axis =1)
def dummy(df):
    sex_dummies = pd.get_dummies(df['Sex'])
    #df['Sex'] = pd.Categorical.from_array(df.Sex).codes
    person_dummies = pd.get_dummies(df['person'])
    #class_dummies = pd.get_dummies(df['Pclass'], prefix= 'Pclass')
    
    #title_dummies = pd.get_dummies(df['Title'])
    df = pd.concat([df, person_dummies, sex_dummies], axis = 1)
    df.drop(['person', 'Sex'], axis = 1, inplace = True)
    #df['Age'] = df.apply(encode_age, axis = 1)
    return df







# ######################################################################
# ######################################################################





   









def make_sub(model_results, PassengerId):
    model_results = [str(int(x)) for x in model_results]
    submission = pd.DataFrame()
    submission['PassengerId'] = PassengerId
    submission['Survived'] = model_results
    submission.set_index(['PassengerId'],inplace=True, drop=True)
    submission.to_csv('titanic_submission.csv')


