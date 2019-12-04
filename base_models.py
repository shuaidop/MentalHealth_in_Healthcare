import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import randint
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import binarize, LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

from sklearn.metrics import roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import RandomizedSearchCV
from statsmodels.stats.proportion import proportions_ztest
import country_list as c
from scipy.stats import pearsonr
train_df = pd.read_csv('survey.csv')


def prop_z_test(value):
    count = np.array([value, 1259-value])
    nobs = np.array([1259, 1259])
    stat, pval = proportions_ztest(count, nobs)
    return('{0:0.3f}'.format(pval))

missing_v_table=pd.DataFrame(train_df.isna().sum()[train_df.isna().sum()>0],columns=['Missing Value'])
missing_v_table['percent']=missing_v_table.apply(lambda x: x/1259)
missing_v_table


train_df = train_df.drop(['comments'], axis= 1)
train_df = train_df.drop(['state'], axis= 1)
train_df = train_df.drop(['Timestamp'], axis= 1)


defaultInt = 0
defaultString = 'NaN'
defaultFloat = 0.0

# Create lists by data tpe
intFeatures = ['Age']###
stringFeatures = ['Gender', 'Country', 'self_employed', 'family_history', 'treatment', 'work_interfere',
                 'no_employees', 'remote_work', 'tech_company', 'anonymity', 'leave', 'mental_health_consequence',
                 'phys_health_consequence', 'coworkers', 'supervisor', 'mental_health_interview', 'phys_health_interview',
                 'mental_vs_physical', 'obs_consequence', 'benefits', 'care_options', 'wellness_program',
                 'seek_help']
floatFeatures = []

# Clean the NaN's
for feature in train_df:
    if feature in intFeatures:
        train_df[feature] = train_df[feature].fillna(defaultInt)
    elif feature in stringFeatures:
        train_df[feature] = train_df[feature].fillna(defaultString)
    elif feature in floatFeatures:
        train_df[feature] = train_df[feature].fillna(defaultFloat)
    else:
        print('Error: Feature %s not recognized.' % feature)



#clean 'Gender'
gender = train_df['Gender'].str.lower()

gender = train_df['Gender'].unique()


male_str = ["male", "m", "male-ish", "maile", "mal", "male (cis)", "make", "male ", "man","msle", "mail", "malr","cis man", "Cis Male", "cis male"]
trans_str = ["trans-female", "something kinda male?", "queer/she/they", "non-binary","nah", "all", "enby", "fluid", "genderqueer", "androgyne", "agender", "male leaning androgynous", "guy (-ish) ^_^", "trans woman", "neuter", "female (trans)", "queer", "ostensibly male, unsure what that really means"]           
female_str = ["cis female", "f", "female", "woman",  "femake", "female ","cis-female/femme", "female (cis)", "femail"]

for (index, row) in train_df.iterrows():

    if str.lower(row.Gender) in male_str:
        train_df['Gender'].replace(to_replace=row.Gender, value='male', inplace=True)

    if str.lower(row.Gender) in female_str:
        train_df['Gender'].replace(to_replace=row.Gender, value='female', inplace=True)

    if str.lower(row.Gender) in trans_str:
        train_df['Gender'].replace(to_replace=row.Gender, value='trans', inplace=True)


stk_list = ['A little about you', 'p']
train_df = train_df[~train_df['Gender'].isin(stk_list)]


over_100_age=list(train_df[train_df.Age>=70].Age.index)
lower_100_age=list(train_df[train_df.Age<=18].Age.index)
for max_ in over_100_age:
    train_df.loc[max_,'Age']=70
for min_ in lower_100_age:
    train_df.loc[min_,'Age']=18


train_df['age_range'] = pd.cut(train_df['Age'], [0,20,30,65,100], labels=["0-20", "21-30", "31-65", "66-100"], include_lowest=True)
train_df['work_interfere'] = train_df['work_interfere'].replace([defaultString], 'Don\'t know' )
train_df['self_employed'] = train_df['self_employed'].replace([defaultString], 'No')

    
train_df['Country_cate']=train_df['Country'].apply(lambda x: c.contry_cat(x.replace(' ','')))



labelDict = {}
for feature in train_df:
    le = preprocessing.LabelEncoder()
    le.fit(train_df[feature])
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    train_df[feature] = le.transform(train_df[feature])
    # Get labels
    labelKey = 'label_' + feature
    labelValue = [*le_name_mapping]
    labelDict[labelKey] =labelValue
    


T_df=pd.DataFrame(corrmat['treatment'])
T_df.columns=['corr']
T_df['name']=T_df.index
T_df['p_value']=T_df.name.apply(lambda x: pearsonr(train_df.treatment,train_df[x])[1])
T_df=T_df.drop(['name'],axis=1)
T_df=T_df.drop(['treatment'],axis=0)
T_df=T_df[T_df.p_value<0.05]
var_list=list(T_df.index)



cate_vars=['Gender', 'benefits', 'care_options', 'wellness_program', 'seek_help', 'anonymity', 'mental_health_consequence', 
 'coworkers', 'mental_health_interview', 'mental_vs_physical']
old_train_df=train_df.copy()

a=train_df.copy()
b=a.drop(cate_vars,axis=1)
train_df=pd.concat([b,pd.get_dummies(a[cate_vars].astype(str))],axis=1)



from sklearn.linear_model import LogisticRegressionCV

lgR = LogisticRegressionCV(random_state=2, solver='lbfgs',multi_class='multinomial',cv=10,max_iter=1000).fit(X_train_old, y_train_old)
print('Logistic Base Line')
print('Accuracy:',accuracy_score(lgR.predict(X_test_old), y_test_old))
print('recall: ',recall_score(lgR.predict(X_test_old),y_test_old))
print('precision: ',precision_score(lgR.predict(X_test_old),y_test_old))

print('           ***** confusion_matrix *****')
print(confusion_matrix(lgR.predict(X_test_old),y_test_old))


lgR_new = LogisticRegressionCV(random_state=2, solver='lbfgs',multi_class='multinomial',cv=10,max_iter=1000).fit(X_train, y_train)
print('Logistic with fewer variables')
print('Accuracy:',accuracy_score(lgR_new.predict(X_test), y_test))
print('recall: ',recall_score(lgR_new.predict(X_test),y_test))
print('precision: ',precision_score(lgR_new.predict(X_test),y_test))


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Number of trees in random forest
n_estimators = [90,100,70]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 3)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_depth': max_depth,
               'criterion': ['gini','entropy']}

clf = RandomForestClassifier(n_jobs=1, random_state=2)
clf.fit(X_train, y_train)
# GridSearchCV(ABC, param_grid=param_grid, scoring = 'roc_auc')
grid = GridSearchCV(clf, random_grid, cv=10, scoring='accuracy')

print('Random Forest Base Model')
print('Accuracy:',accuracy_score(clf.predict(X_test), y_test))
print('recall: ',recall_score(clf.predict(X_test),y_test))
print('precision: ',precision_score(clf.predict(X_test),y_test))



print('Random Forest Tunned Model')
grid.fit(X_train, y_train)
final_rf=grid.best_estimator_
print('Accuracy:',accuracy_score(final_rf.predict(X_test), y_test))
print('recall: ',recall_score(final_rf.predict(X_test),y_test))
print('precision: ',precision_score(final_rf.predict(X_test),y_test))




from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier(random_state=2,n_estimators=30)
model.fit(X_train, y_train)
print('AdaBoost Classifier')
print('Accuracy:',accuracy_score(model.predict(X_test), y_test))
print('recall: ',recall_score(model.predict(X_test),y_test))
print('precision: ',precision_score(model.predict(X_test),y_test))



random_grid = {"learning_rate":[1,1.2,1.4],
              "n_estimators" :   [26,30,36]
             }



print('Adaboost Tunned Model')
gridAda = GridSearchCV(model, random_grid, cv=10, scoring='accuracy')
gridAda.fit(X_train, y_train)
final_ada=gridAda.best_estimator_
print('Accuracy:',accuracy_score(final_ada.predict(X_test), y_test))
print('recall: ',recall_score(final_ada.predict(X_test),y_test))
print('precision: ',precision_score(final_ada.predict(X_test),y_test))




from sklearn.ensemble import VotingClassifier
estimators=[('Ada',final_ada ), ('rf', final_rf), ('log_reg', lgR_new)]
#create our voting classifier, inputting our models
ensemble = VotingClassifier(estimators, voting='soft')
ensemble.fit(X_train,y_train)


print('ensemble moddles')
print('Accuracy:',accuracy_score(ensemble.predict(X_test), y_test))
print('recall: ',recall_score(ensemble.predict(X_test),y_test))
print('precision: ',precision_score(ensemble.predict(X_test),y_test))



