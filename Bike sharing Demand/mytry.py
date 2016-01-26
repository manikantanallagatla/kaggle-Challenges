import pandas as pd
from sklearn import ensemble
# The competition datafiles are in the directory ../input
file_train = "train.csv"
file_test = "test.csv"
titanic = pd.read_csv(file_train)
titanic_test = pd.read_csv(file_test)



titanic['season'] = titanic['season'].fillna(titanic['season'].median())
titanic['holiday'] = titanic['holiday'].fillna(titanic['holiday'].median())
titanic['workingday'] = titanic['workingday'].fillna(titanic['workingday'].median())
titanic['weather'] = titanic['weather'].fillna(titanic['weather'].median())
titanic['temp'] = titanic['temp'].fillna(titanic['temp'].median())
titanic['atemp'] = titanic['atemp'].fillna(titanic['atemp'].median())
titanic['humidity'] = titanic['humidity'].fillna(titanic['humidity'].median())
titanic['windspeed'] = titanic['windspeed'].fillna(titanic['windspeed'].median())
titanic['registered'] = titanic['registered'].fillna(titanic['registered'].median())
titanic['casual'] = titanic['casual'].fillna(titanic['casual'].median())
titanic['count'] = titanic['count'].fillna(titanic['count'].median())

titanic_test['season'] = titanic_test['season'].fillna(titanic_test['season'].median())
titanic_test['holiday'] = titanic_test['holiday'].fillna(titanic_test['holiday'].median())
titanic_test['workingday'] = titanic_test['workingday'].fillna(titanic_test['workingday'].median())
titanic_test['weather'] = titanic_test['weather'].fillna(titanic_test['weather'].median())
titanic_test['temp'] = titanic_test['temp'].fillna(titanic_test['temp'].median())
titanic_test['atemp'] = titanic_test['atemp'].fillna(titanic_test['atemp'].median())
titanic_test['humidity'] = titanic_test['humidity'].fillna(titanic_test['humidity'].median())
titanic_test['windspeed'] = titanic_test['windspeed'].fillna(titanic_test['windspeed'].median())



titanic.to_csv("train1.csv", index=False)
titanic_test.to_csv("test1.csv", index=False)
