import pandas as pd
from sklearn import ensemble
# The competition datafiles are in the directory ../input
file_train = "train.csv"
file_test = "test.csv"
titanic = pd.read_csv(file_train)
titanic_test = pd.read_csv(file_test)

titanic['Item_Weight'] = titanic['Item_Weight'].fillna(titanic['Item_Weight'].median())
titanic.loc[titanic["Item_Fat_Content"] == "Low Fat", "Item_Fat_Content"] = 1
titanic.loc[titanic["Item_Fat_Content"] == "Regular", "Item_Fat_Content"] = 2
titanic.loc[titanic["Item_Fat_Content"] == "LF", "Item_Fat_Content"] = 1
titanic.loc[titanic["Item_Fat_Content"] == "reg", "Item_Fat_Content"] = 2
titanic.loc[titanic["Item_Fat_Content"] == "low fat", "Item_Fat_Content"] = 1
titanic['Item_Fat_Content'] = titanic['Item_Fat_Content'].fillna(titanic['Item_Fat_Content'].median())
titanic['Item_Visibility'] = titanic['Item_Visibility'].fillna(titanic['Item_Visibility'].median())
titanic.loc[titanic["Item_Type"] == "Dairy", "Item_Type"] = 1
titanic.loc[titanic["Item_Type"] == "Soft Drinks", "Item_Type"] = 2
titanic.loc[titanic["Item_Type"] == "Meat", "Item_Type"] = 3
titanic.loc[titanic["Item_Type"] == "Fruits and Vegetables", "Item_Type"] = 4
titanic.loc[titanic["Item_Type"] == "Household", "Item_Type"] = 5
titanic.loc[titanic["Item_Type"] == "Baking Goods", "Item_Type"] = 6
titanic.loc[titanic["Item_Type"] == "Snack Foods", "Item_Type"] = 7
titanic.loc[titanic["Item_Type"] == "Frozen Foods", "Item_Type"] = 8
titanic.loc[titanic["Item_Type"] == "Breakfast", "Item_Type"] = 9
titanic.loc[titanic["Item_Type"] == "Health and Hygiene", "Item_Type"] = 10
titanic.loc[titanic["Item_Type"] == "Hard Drinks", "Item_Type"] = 11
titanic.loc[titanic["Item_Type"] == "Canned", "Item_Type"] = 12
titanic.loc[titanic["Item_Type"] == "Breads", "Item_Type"] = 13
titanic.loc[titanic["Item_Type"] == "Others", "Item_Type"] = 14
titanic.loc[titanic["Item_Type"] == "Seafood", "Item_Type"] = 15
titanic.loc[titanic["Item_Type"] == "Starchy Foods", "Item_Type"] = 16
titanic['Item_Type'] = titanic['Item_Type'].fillna(titanic['Item_Type'].median())
titanic['Item_MRP'] = titanic['Item_MRP'].fillna(titanic['Item_MRP'].median())
titanic['Outlet_Establishment_Year'] = titanic['Outlet_Establishment_Year'].fillna(titanic['Outlet_Establishment_Year'].median())
titanic.loc[titanic["Outlet_Size"] == "Medium", "Outlet_Size"] = 1
titanic.loc[titanic["Outlet_Size"] == "Small", "Outlet_Size"] = 2
titanic.loc[titanic["Outlet_Size"] == "High", "Outlet_Size"] = 3
titanic['Outlet_Size'] = titanic['Outlet_Size'].fillna(titanic['Outlet_Size'].median())
titanic.loc[titanic["Outlet_Location_Type"] == "Tier 1", "Outlet_Location_Type"] = 1
titanic.loc[titanic["Outlet_Location_Type"] == "Tier 2", "Outlet_Location_Type"] = 2
titanic.loc[titanic["Outlet_Location_Type"] == "Tier 3", "Outlet_Location_Type"] = 3
titanic['Outlet_Location_Type'] = titanic['Outlet_Location_Type'].fillna(titanic['Outlet_Location_Type'].median())
titanic.loc[titanic["Outlet_Type"] == "Supermarket Type1", "Outlet_Type"] = 1
titanic.loc[titanic["Outlet_Type"] == "Grocery Store", "Outlet_Type"] = 2
titanic.loc[titanic["Outlet_Type"] == "Supermarket Type2", "Outlet_Type"] = 3
titanic.loc[titanic["Outlet_Type"] == "Supermarket Type3", "Outlet_Type"] = 4
titanic['Outlet_Type'] = titanic['Outlet_Type'].fillna(titanic['Outlet_Type'].median())
titanic['Item_Outlet_Sales'] = titanic['Item_Outlet_Sales'].fillna(titanic['Item_Outlet_Sales'].median())

titanic_test['Item_Weight'] = titanic_test['Item_Weight'].fillna(titanic_test['Item_Weight'].median())
titanic_test.loc[titanic_test["Item_Fat_Content"] == "Low Fat", "Item_Fat_Content"] = 1
titanic_test.loc[titanic_test["Item_Fat_Content"] == "Regular", "Item_Fat_Content"] = 2
titanic_test.loc[titanic_test["Item_Fat_Content"] == "LF", "Item_Fat_Content"] = 1
titanic_test.loc[titanic_test["Item_Fat_Content"] == "reg", "Item_Fat_Content"] = 2
titanic_test.loc[titanic_test["Item_Fat_Content"] == "low fat", "Item_Fat_Content"] = 1
titanic_test['Item_Fat_Content'] = titanic_test['Item_Fat_Content'].fillna(titanic_test['Item_Fat_Content'].median())
titanic_test['Item_Visibility'] = titanic_test['Item_Visibility'].fillna(titanic_test['Item_Visibility'].median())
titanic_test.loc[titanic_test["Item_Type"] == "Dairy", "Item_Type"] = 1
titanic_test.loc[titanic_test["Item_Type"] == "Soft Drinks", "Item_Type"] = 2
titanic_test.loc[titanic_test["Item_Type"] == "Meat", "Item_Type"] = 3
titanic_test.loc[titanic_test["Item_Type"] == "Fruits and Vegetables", "Item_Type"] = 4
titanic_test.loc[titanic_test["Item_Type"] == "Household", "Item_Type"] = 5
titanic_test.loc[titanic_test["Item_Type"] == "Baking Goods", "Item_Type"] = 6
titanic_test.loc[titanic_test["Item_Type"] == "Snack Foods", "Item_Type"] = 7
titanic_test.loc[titanic_test["Item_Type"] == "Frozen Foods", "Item_Type"] = 8
titanic_test.loc[titanic_test["Item_Type"] == "Breakfast", "Item_Type"] = 9
titanic_test.loc[titanic_test["Item_Type"] == "Health and Hygiene", "Item_Type"] = 10
titanic_test.loc[titanic_test["Item_Type"] == "Hard Drinks", "Item_Type"] = 11
titanic_test.loc[titanic_test["Item_Type"] == "Canned", "Item_Type"] = 12
titanic_test.loc[titanic_test["Item_Type"] == "Breads", "Item_Type"] = 13
titanic_test.loc[titanic_test["Item_Type"] == "Others", "Item_Type"] = 14
titanic_test.loc[titanic_test["Item_Type"] == "Seafood", "Item_Type"] = 15
titanic_test.loc[titanic_test["Item_Type"] == "Starchy Foods", "Item_Type"] = 16
titanic_test['Item_Type'] = titanic_test['Item_Type'].fillna(titanic_test['Item_Type'].median())
titanic_test['Item_MRP'] = titanic_test['Item_MRP'].fillna(titanic_test['Item_MRP'].median())
titanic_test['Outlet_Establishment_Year'] = titanic_test['Outlet_Establishment_Year'].fillna(titanic_test['Outlet_Establishment_Year'].median())
titanic_test.loc[titanic_test["Outlet_Size"] == "Medium", "Outlet_Size"] = 1
titanic_test.loc[titanic_test["Outlet_Size"] == "Small", "Outlet_Size"] = 2
titanic_test.loc[titanic_test["Outlet_Size"] == "High", "Outlet_Size"] = 3
titanic_test['Outlet_Size'] = titanic_test['Outlet_Size'].fillna(titanic_test['Outlet_Size'].median())
titanic_test.loc[titanic_test["Outlet_Location_Type"] == "Tier 1", "Outlet_Location_Type"] = 1
titanic_test.loc[titanic_test["Outlet_Location_Type"] == "Tier 2", "Outlet_Location_Type"] = 2
titanic_test.loc[titanic_test["Outlet_Location_Type"] == "Tier 3", "Outlet_Location_Type"] = 3
titanic_test['Outlet_Location_Type'] = titanic_test['Outlet_Location_Type'].fillna(titanic_test['Outlet_Location_Type'].median())
titanic_test.loc[titanic_test["Outlet_Type"] == "Supermarket Type1", "Outlet_Type"] = 1
titanic_test.loc[titanic_test["Outlet_Type"] == "Grocery Store", "Outlet_Type"] = 2
titanic_test.loc[titanic_test["Outlet_Type"] == "Supermarket Type2", "Outlet_Type"] = 3
titanic_test.loc[titanic_test["Outlet_Type"] == "Supermarket Type3", "Outlet_Type"] = 4
titanic_test['Outlet_Type'] = titanic_test['Outlet_Type'].fillna(titanic_test['Outlet_Type'].median())

titanic.to_csv("train1.csv", index=False)
titanic_test.to_csv("test1.csv", index=False)
titanic.head()
feature_cols = [col for col in titanic.columns if col not in ['Item_Identifier','Outlet_Identifier','Item_Outlet_Sales']]

X_train = titanic[feature_cols]
X_test = titanic_test[feature_cols]
y = titanic['Item_Outlet_Sales'] # target
test_ids = titanic_test['Item_Identifier'] # for submission
test_outletids = titanic_test['Outlet_Identifier'] # for submission
clf = ensemble.RandomForestClassifier(n_estimators=200,n_jobs=-1,random_state=0)
clf.fit(X_train, y)
file_submission = "kaggle.csv"
with open(file_submission, "w") as outfile:
    outfile.write("Item_Identifier,Outlet_Identifier,Item_Outlet_Sales\n")
    for e, val in enumerate(list(clf.predict(X_test))):
        outfile.write("%s,%s,%s\n"%(test_ids[e],test_outletids[e],val))