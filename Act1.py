#ML - Machine Learning
# Steps:
# 1. Gathering the data and load the data into the program
# 2. Data Processing - prepare the data for further machine learning projects. Remove/replace null values or objective data -> numerical values
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
