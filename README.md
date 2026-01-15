# Task-1-AIML
CODE:1)Titanic.csv
import pandas as pd

df = pd.read_csv("titanic.csv")
df.head()   
df.tail()   
df.info()
df.describe()
df.isnull().sum()
df.shape
OBSERVATION:- 
Number of columns

What kind of information each row represents (one passenger / one student)
Numerical

Age, Fare, SibSp, Parch

Categorical

Sex, Embarked, Cabin, Ticket

Binary

Survived (0/1)

Ordinal (if any)

Pclass (1st, 2nd, 3rd class → ordered)
Mean age

Fare range

Presence of outliers
Age has missing values

Cabin has many missing values

Data cleaning required
Number of rows and columns

Dataset is suitable for classification ML problems

Requires preprocessing (missing values, encoding)

CODE:2)StudentsPerformance.csv
**# ================================
# Task 1: Understanding Dataset & Data Types
# Student Performance Dataset
# ================================

import pandas as pd

# STEP 1: Load the dataset
df = pd.read_csv("StudentsPerformance.csv")

print("\nSTEP 1: First 5 Records")
print(df.head())

print("\nSTEP 1: Last 5 Records")
print(df.tail())


# STEP 2: Dataset structure
print("\nSTEP 2: Dataset Information")
print(df.info())


# STEP 3: Statistical summary
print("\nSTEP 3: Statistical Summary")
print(df.describe())


# STEP 4: Identify data types manually
numerical_features = [
    'math score',
    'reading score',
    'writing score'
]

categorical_features = [
    'gender',
    'race/ethnicity',
    'parental level of education',
    'lunch',
    'test preparation course'
]

print("\nSTEP 4: Numerical Features")
print(numerical_features)

print("\nSTEP 4: Categorical Features")
print(categorical_features)


# STEP 5: Check missing values
print("\nSTEP 5: Missing Values in Each Column")
print(df.isnull().sum())


# STEP 6: Unique values & distribution of categorical columns
print("\nSTEP 6: Categorical Data Distribution")

for col in categorical_features:
    print(f"\nColumn: {col}")
    print(df[col].value_counts())


# STEP 7: Target and input features
target_variable = 'math score'
input_features = [col for col in df.columns if col != target_variable]

print("\nSTEP 7: Target Variable")
print(target_variable)

print("\nSTEP 7: Input Features")
print(input_features)


# STEP 8: Dataset size & ML suitability
print("\nSTEP 8: Dataset Shape (Rows, Columns)")
print(df.shape)

print("\nDataset is suitable for Machine Learning.")
print("Requires encoding of categorical features before modeling.")

