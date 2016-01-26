import pandas as pd
from sklearn import ensemble
# The competition datafiles are in the directory ../input
file_train = "train.csv"
file_test = "test.csv"
titanic = pd.read_csv(file_train)
titanic_test = pd.read_csv(file_test)

titanic.loc[titanic["Var1"] == "A", "Var1"] = 1
titanic.loc[titanic["Var1"] == "B", "Var1"] = 2
titanic.loc[titanic["Var1"] == "C", "Var1"] = 3
titanic.loc[titanic["Var1"] == "D", "Var1"] = 4
titanic.loc[titanic["Var1"] == "E", "Var1"] = 5
titanic.loc[titanic["Var1"] == "F", "Var1"] = 6
titanic.loc[titanic["Var1"] == "G", "Var1"] = 7
titanic['Var1'] = titanic['Var1'].fillna(titanic['Var1'].median())
titanic.loc[titanic["WorkStatus"] == "keeping house", "WorkStatus"] = 1
titanic.loc[titanic["WorkStatus"] == "working fulltime", "WorkStatus"] = 2
titanic.loc[titanic["WorkStatus"] == "retired", "WorkStatus"] = 3
titanic.loc[titanic["WorkStatus"] == "working parttime", "WorkStatus"] = 4
titanic.loc[titanic["WorkStatus"] == "other", "WorkStatus"] = 5
titanic.loc[titanic["WorkStatus"] == "temp not working", "WorkStatus"] = 6
titanic.loc[titanic["WorkStatus"] == "school", "WorkStatus"] = 7
titanic.loc[titanic["WorkStatus"] == "unempl, laid off", "WorkStatus"] = 8
titanic['WorkStatus'] = titanic['WorkStatus'].fillna(titanic['WorkStatus'].median())
titanic['Score'] = titanic['Score'].fillna(titanic['Score'].median())
titanic.loc[titanic["Divorce"] == "yes", "Divorce"] = 1
titanic.loc[titanic["Divorce"] == "no", "Divorce"] = 2
titanic['Divorce'] = titanic['Divorce'].fillna(titanic['Divorce'].median())
titanic.loc[titanic["Widowed"] == "yes", "Widowed"] = 1
titanic.loc[titanic["Widowed"] == "no", "Widowed"] = 2
titanic['Widowed'] = titanic['Widowed'].fillna(titanic['Widowed'].median())
titanic['Education'] = titanic['Education'].fillna(titanic['Education'].median())
titanic.loc[titanic["Residence_Region"] == "middle atlantic", "Residence_Region"] = 1
titanic.loc[titanic["Residence_Region"] == "foreign", "Residence_Region"] = 2
titanic.loc[titanic["Residence_Region"] == "south atlantic", "Residence_Region"] = 3
titanic.loc[titanic["Residence_Region"] == "e. nor. central", "Residence_Region"] = 4
titanic.loc[titanic["Residence_Region"] == "new england", "Residence_Region"] = 5
titanic.loc[titanic["Residence_Region"] == "w. sou. central", "Residence_Region"] = 6
titanic.loc[titanic["Residence_Region"] == "e. sou. central", "Residence_Region"] = 7
titanic.loc[titanic["Residence_Region"] == "pacific", "Residence_Region"] = 8
titanic.loc[titanic["Residence_Region"] == "w. nor. central", "Residence_Region"] = 9
titanic.loc[titanic["Residence_Region"] == "mountain", "Residence_Region"] = 10
titanic['Residence_Region'] = titanic['Residence_Region'].fillna(titanic['Residence_Region'].median())
titanic['babies'] = titanic['babies'].fillna(titanic['babies'].median())
titanic['preteen'] = titanic['preteen'].fillna(titanic['preteen'].median())
titanic['teens'] = titanic['teens'].fillna(titanic['teens'].median())
titanic.loc[titanic["income"] == "$10000 - 14999", "income"] = 1
titanic.loc[titanic["income"] == "$15000 - 19999", "income"] = 2
titanic.loc[titanic["income"] == "$25000 or more", "income"] = 3
titanic.loc[titanic["income"] == "$8000 to 9999", "income"] = 4
titanic.loc[titanic["income"] == "$5000 to 5999", "income"] = 5
titanic.loc[titanic["income"] == "$20000 - 24999", "income"] = 6
titanic.loc[titanic["income"] == "$7000 to 7999", "income"] = 7
titanic.loc[titanic["income"] == "$6000 to 6999", "income"] = 8
titanic.loc[titanic["income"] == "$3000 to 3999", "income"] = 9
titanic.loc[titanic["income"] == "$1000 to 2999", "income"] = 10
titanic.loc[titanic["income"] == "lt $1000", "income"] = 11
titanic.loc[titanic["income"] == "$4000 to 4999", "income"] = 12
titanic['income'] = titanic['income'].fillna(titanic['income'].median())
titanic.loc[titanic["Engagement_Religion"] == "sevrl times a yr", "Engagement_Religion"] = 1
titanic.loc[titanic["Engagement_Religion"] == "more thn once wk", "Engagement_Religion"] = 2
titanic.loc[titanic["Engagement_Religion"] == "once a year", "Engagement_Religion"] = 3
titanic.loc[titanic["Engagement_Religion"] == "never", "Engagement_Religion"] = 4
titanic.loc[titanic["Engagement_Religion"] == "2-3x a month", "Engagement_Religion"] = 5
titanic.loc[titanic["Engagement_Religion"] == "lt once a year", "Engagement_Religion"] = 3
titanic.loc[titanic["Engagement_Religion"] == "once a month", "Engagement_Religion"] = 7
titanic.loc[titanic["Engagement_Religion"] == "nrly every week", "Engagement_Religion"] = 8
titanic.loc[titanic["Engagement_Religion"] == "every week", "Engagement_Religion"] = 8
titanic.loc[titanic["Engagement_Religion"] == "sevrl times a yr", "Engagement_Religion"] = 9
#titanic.loc[titanic["Engagement_Religion"] == "every week", "Engagement_Religion"] = 8
#titanic.loc[titanic["Engagement_Religion"] == "every week", "Engagement_Religion"] = 8
titanic['Engagement_Religion'] = titanic['Engagement_Religion'].fillna(titanic['Engagement_Religion'].median())
titanic['Var2'] = titanic['Var2'].fillna(titanic['Var2'].median())
titanic['TVhours'] = titanic['TVhours'].fillna(titanic['TVhours'].median())
titanic['Gender'] = titanic['Gender'].fillna(titanic['Gender'].median())
titanic['Unemployed10'] = titanic['Unemployed10'].fillna(titanic['Unemployed10'].median())


titanic_test.loc[titanic_test["Var1"] == "A", "Var1"] = 1
titanic_test.loc[titanic_test["Var1"] == "B", "Var1"] = 2
titanic_test.loc[titanic_test["Var1"] == "C", "Var1"] = 3
titanic_test.loc[titanic_test["Var1"] == "D", "Var1"] = 4
titanic_test.loc[titanic_test["Var1"] == "E", "Var1"] = 5
titanic_test.loc[titanic_test["Var1"] == "F", "Var1"] = 6
titanic_test.loc[titanic_test["Var1"] == "G", "Var1"] = 7
titanic_test['Var1'] = titanic_test['Var1'].fillna(titanic_test['Var1'].median())
titanic_test.loc[titanic_test["WorkStatus"] == "keeping house", "WorkStatus"] = 1
titanic_test.loc[titanic_test["WorkStatus"] == "working fulltime", "WorkStatus"] = 2
titanic_test.loc[titanic_test["WorkStatus"] == "retired", "WorkStatus"] = 3
titanic_test.loc[titanic_test["WorkStatus"] == "working parttime", "WorkStatus"] = 4
titanic_test.loc[titanic_test["WorkStatus"] == "other", "WorkStatus"] = 5
titanic_test.loc[titanic_test["WorkStatus"] == "temp not working", "WorkStatus"] = 6
titanic_test.loc[titanic_test["WorkStatus"] == "school", "WorkStatus"] = 7
titanic_test.loc[titanic_test["WorkStatus"] == "unempl, laid off", "WorkStatus"] = 8
titanic_test['WorkStatus'] = titanic_test['WorkStatus'].fillna(titanic_test['WorkStatus'].median())
titanic_test['Score'] = titanic_test['Score'].fillna(titanic_test['Score'].median())
titanic_test.loc[titanic_test["Divorce"] == "yes", "Divorce"] = 1
titanic_test.loc[titanic_test["Divorce"] == "no", "Divorce"] = 2
titanic_test['Divorce'] = titanic_test['Divorce'].fillna(titanic_test['Divorce'].median())
titanic_test.loc[titanic_test["Widowed"] == "yes", "Widowed"] = 1
titanic_test.loc[titanic_test["Widowed"] == "no", "Widowed"] = 2
titanic_test['Widowed'] = titanic_test['Widowed'].fillna(titanic_test['Widowed'].median())
titanic_test['Education'] = titanic_test['Education'].fillna(titanic_test['Education'].median())
titanic_test.loc[titanic_test["Residence_Region"] == "middle atlantic", "Residence_Region"] = 1
titanic_test.loc[titanic_test["Residence_Region"] == "foreign", "Residence_Region"] = 2
titanic_test.loc[titanic_test["Residence_Region"] == "south atlantic", "Residence_Region"] = 3
titanic_test.loc[titanic_test["Residence_Region"] == "e. nor. central", "Residence_Region"] = 4
titanic_test.loc[titanic_test["Residence_Region"] == "new england", "Residence_Region"] = 5
titanic_test.loc[titanic_test["Residence_Region"] == "w. sou. central", "Residence_Region"] = 6
titanic_test.loc[titanic_test["Residence_Region"] == "e. sou. central", "Residence_Region"] = 7
titanic_test.loc[titanic_test["Residence_Region"] == "pacific", "Residence_Region"] = 8
titanic_test.loc[titanic_test["Residence_Region"] == "w. nor. central", "Residence_Region"] = 9
titanic_test.loc[titanic_test["Residence_Region"] == "mountain", "Residence_Region"] = 10
titanic_test['Residence_Region'] = titanic_test['Residence_Region'].fillna(titanic_test['Residence_Region'].median())
titanic_test['babies'] = titanic_test['babies'].fillna(titanic_test['babies'].median())
titanic_test['preteen'] = titanic_test['preteen'].fillna(titanic_test['preteen'].median())
titanic_test['teens'] = titanic_test['teens'].fillna(titanic_test['teens'].median())
titanic_test.loc[titanic_test["income"] == "$10000 - 14999", "income"] = 1
titanic_test.loc[titanic_test["income"] == "$15000 - 19999", "income"] = 2
titanic_test.loc[titanic_test["income"] == "$25000 or more", "income"] = 3
titanic_test.loc[titanic_test["income"] == "$8000 to 9999", "income"] = 4
titanic_test.loc[titanic_test["income"] == "$5000 to 5999", "income"] = 5
titanic_test.loc[titanic_test["income"] == "$20000 - 24999", "income"] = 6
titanic_test.loc[titanic_test["income"] == "$7000 to 7999", "income"] = 7
titanic_test.loc[titanic_test["income"] == "$6000 to 6999", "income"] = 8
titanic_test.loc[titanic_test["income"] == "$3000 to 3999", "income"] = 9
titanic_test.loc[titanic_test["income"] == "$1000 to 2999", "income"] = 10
titanic_test.loc[titanic_test["income"] == "lt $1000", "income"] = 11
titanic_test.loc[titanic_test["income"] == "$4000 to 4999", "income"] = 12
titanic_test['income'] = titanic_test['income'].fillna(titanic_test['income'].median())
titanic_test.loc[titanic_test["Engagement_Religion"] == "sevrl times a yr", "Engagement_Religion"] = 1
titanic_test.loc[titanic_test["Engagement_Religion"] == "more thn once wk", "Engagement_Religion"] = 2
titanic_test.loc[titanic_test["Engagement_Religion"] == "once a year", "Engagement_Religion"] = 3
titanic_test.loc[titanic_test["Engagement_Religion"] == "never", "Engagement_Religion"] = 4
titanic_test.loc[titanic_test["Engagement_Religion"] == "2-3x a month", "Engagement_Religion"] = 5
titanic_test.loc[titanic_test["Engagement_Religion"] == "lt once a year", "Engagement_Religion"] = 3
titanic_test.loc[titanic_test["Engagement_Religion"] == "once a month", "Engagement_Religion"] = 7
titanic_test.loc[titanic_test["Engagement_Religion"] == "nrly every week", "Engagement_Religion"] = 8
titanic_test.loc[titanic_test["Engagement_Religion"] == "every week", "Engagement_Religion"] = 8
titanic_test.loc[titanic_test["Engagement_Religion"] == "sevrl times a yr", "Engagement_Religion"] = 9
#titanic_test.loc[titanic_test["Engagement_Religion"] == "every week", "Engagement_Religion"] = 8
#titanic_test.loc[titanic_test["Engagement_Religion"] == "every week", "Engagement_Religion"] = 8
titanic_test['Engagement_Religion'] = titanic_test['Engagement_Religion'].fillna(titanic_test['Engagement_Religion'].median())
titanic_test['Var2'] = titanic_test['Var2'].fillna(titanic_test['Var2'].median())
titanic_test['TVhours'] = titanic_test['TVhours'].fillna(titanic_test['TVhours'].median())
titanic_test['Gender'] = titanic_test['Gender'].fillna(titanic_test['Gender'].median())
titanic_test['Unemployed10'] = titanic_test['Unemployed10'].fillna(titanic_test['Unemployed10'].median())


#titanic.to_csv("train1.csv", index=False)
#titanic_test.to_csv("test1.csv", index=False)
#titanic.head()
feature_cols = [col for col in titanic.columns if col not in ['ID','Happy']]

X_train = titanic[feature_cols]
X_test = titanic_test[feature_cols]
y = titanic['Happy'] # target
test_ids = titanic_test['ID'] # for submission
clf = ensemble.RandomForestClassifier(n_estimators=200,n_jobs=-1,random_state=0)
clf.fit(X_train, y)
file_submission = "kaggle.csv"
with open(file_submission, "w") as outfile:
    outfile.write("ID,Happy\n")
    for e, val in enumerate(list(clf.predict(X_test))):
        outfile.write("%s,%s\n"%(test_ids[e],val))