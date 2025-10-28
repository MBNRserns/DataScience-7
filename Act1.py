#ML - Machine Learning
# Steps:
# 1. Gathering the data and load the data into the program
# 2. Data Processing - prepare the data for further machine learning projects. Remove/replace null values or objective data(string) -> numerical values
# 3. Data Analysis - Analyse the data so that you can define your input and output columns and drop the rest. Whats x and whats y
# 4. Define Input and Output - Create separate dataframes for input - output
# 5. Perform the test-train split - Split up the data into testing and training data
# 6. Select the machine learning algorithm perform the training of the model
# 7. Perform predictions and compare the predictions with actual result to calculate accuracy

import numpy as np
import pandas as pd

#Step 1
dataset = pd.read_csv("Data.csv")
print(dataset.info())

#Step 3,4
X= dataset.iloc[:, :-1].values #picked all except last column
Y= dataset.iloc[:, -1].values #picked last column
print(X)
print(Y)

#Step 2
#Replace null values with mean
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])
print(X)

#Converting Objective data to Numerical Values
#Purchased Column
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
Y=encoder.fit_transform(Y)
print(Y)

#Country Column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[0])], remainder='passthrough')
X = pd.DataFrame(ct.fit_transform(X))
print(X)

#Step 5
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size= 0.2, random_state= 1)
print("X_train\n",X_train)
print("X_test\n",X_test)
print("Y_train\n",Y_train)
print("Y_test\n",Y_test)

#Step 6
#Scaling data down from -1 to +1
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train.iloc[:, 1:3] = scaler.fit_transform(X_train.iloc[:, 1:3])
X_test.iloc[:, 1:3] = scaler.fit_transform(X_test.iloc[:, 1:3])
print()
print(X_train)
print()
print(X_test)