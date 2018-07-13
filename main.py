import gen_feature  as gf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np

# generate new feature 'perishing_mother_wife'
# replace ngsGBC.best_estimator_ame with surname
def process_surname(nm):
    return nm.split(',')[0].lower()

def perishing_mother_wife(passenger): 
    surname, Pclass, person = passenger
    return 1.0 if (surname in perishing_female_surnames) else 0.0

def gen_perishing_mother_wife(df):
    df['perishing_mother_wife'] = df[['surname', 'Pclass', 'person']].apply(perishing_mother_wife, axis=1)




# generate new feature 'surviving_father_husband'
#### Survivng Males

def surviving_father_husband(passenger): 

    surname, Pclass, person = passenger
    return 1.0 if (surname in surviving_male_surnames) else 0.0
def gen_surviving_father_husband(df):
    
    df['surviving_father_husband'] = df[['surname', 'Pclass', 'person']].apply(surviving_father_husband, axis=1)



if __name__=="__main__":
    data_train = pd.read_csv("train.csv")
    data_test = pd.read_csv("test.csv")
    full_data = pd.concat([data_train, data_test], axis=0)
    #fill the missing 'Embarked' & 'Fare' value
    full_data.Embarked.fillna('S', inplace=True)
    full_data.Fare.fillna(np.median(full_data.Fare[full_data.Fare.notnull()]), inplace=True)
    #full_data['Age'] = full_data['Age'].fillna(0)
    #full_data["Fare"] = full_data["Fare"].map(lambda i: np.log(i) if i > 0 else 0)
    #full_data.Fare.fillna(np.median(full_data.Fare[full_data.Fare.notnull()]), inplace=True)
    # feature engineering
    full_data['Title'] = full_data["Name"].apply(gf.get_title)
    
    #gf.convert_name(full_data)
    #gf.set_missing_ages_title(full_data)

    gf.convert_to_type(full_data)
    gf.get_family_size(full_data)
    gf.generate_family_id(full_data)
    
    gf.generate_person_label(full_data)

    full_data['surname'] = full_data['Name'].apply(process_surname)
    perishing_female_surnames = list(set(full_data[(full_data.person == 'female_adult') &
        (full_data.Survived == 0.0) &
        ((full_data.Parch > 0) | (full_data.SibSp > 0))]['surname'].values))
    surviving_male_surnames = list(set(full_data[(full_data.person == 'male_adult') &
                                        (full_data.Survived == 1.0) &
                                        ((full_data.Parch > 0) | (full_data.SibSp > 0))]['surname'].values))
    gen_perishing_mother_wife(full_data)
    gen_surviving_father_husband(full_data)
    #gf.gen_first_class_male(full_data)
    print(full_data[full_data.Age.isnull()])
    full_data = gf.dummy(full_data)
    
    print(full_data.columns)
    gf.pred_age(full_data)
    
    full_data.drop(['surname', 'Name', 'Title','Cabin','Embarked'], axis = 1, inplace = True)
    
    X_t_data = full_data[(full_data.Survived.isnull() == True)]
    X_test = X_t_data.drop(['Survived', 'PassengerId'], axis = 1)
    X_data = full_data[full_data.Survived.isnull() == False]
    X_train = X_data.drop(['Survived', 'PassengerId'], axis = 1)
    y_train = X_data.loc[:,'Survived']
    
    # building models
    # print('building GradientBoostingClassifier ...')
    # gbc_model, gbc_score = build_model_gbc(X_train, y_train)
    # print('best score for GradientBoostingClassifier', gbc_score)
    # print('parameters', gbc_model.get_params())

    # print('building RandomForestClassifier ...')
    # rfc_model, rfc_score = build_model_rfc(X_train, y_train)
    # print('best score for RandomForestClassifier', rfc_score)
    # print('parameters', rfc_model.get_params())
    
    
    #lrc = LogisticRegression()
    print(X_train.columns)
    X_training,X_val, y_training, y_val =\
        train_test_split(X_train,y_train,test_size=0.2, random_state=0)
    rfc_model = RandomForestClassifier(n_estimators = 300, min_samples_leaf = 4, class_weight={0:0.745,1:0.255})
   
    rfc_model.fit(X_training, y_training)
    val_rfc = rfc_model.predict(X_val)  
    print(metrics.classification_report(y_val, val_rfc))
    val_rfc = pd.DataFrame(val_rfc, columns=['Predicted']) 
    X_val = X_val.reset_index(drop = True)
    y_val = y_val.reset_index(drop = True)
    y_val = y_val.to_frame()
    y_val.columns = ['Truth']
    X_val_concat = pd.concat([X_val,y_val,val_rfc], axis = 1)
    bad_case = X_val_concat[X_val_concat.Truth != X_val_concat.Predicted]
    #print(bad_case.iloc[:,18:26])

    
    #pred_gbc = gbc_model.predict(X_test)
    pred_rfc = rfc_model.predict(X_test)
    gf.make_sub(pred_rfc, X_t_data['PassengerId'])