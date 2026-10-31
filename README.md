<H3>ENTER YOUR NAME: Arunmozhi Varman T</H3>
<H3>ENTER YOUR REGISTER NO: 212223230022</H3>
<H3>EX. NO.1</H3>
<H3>DATE: 31-10-2026</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:

### Import Libraries
```
import pandas as pd
import seaborn as sns
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy import stats
import numpy as np
```
### Read the dataset
```
df = pd.read_csv('Churn_Modelling.csv')
print(df)
```
### values of X & Y
```
X = df.iloc[:, :-1].values
print(X)
y = df.iloc[:, -1].values
print(y)
```
### Check for outliers
```
df.describe()
```
### Missing Values
```
print(df.isnull().sum())
df.fillna(df.mean().round(1), inplace=True)
print(df.isnull().sum())
y = df.iloc[:, -1].values
print(y)
```
### Check for duplicates
```
df.duplicated()
```
### Droppimg string values from dataset
```
data = df.drop(['Surname', 'Geography','Gender'], axis=1)
```
### Normalize dataset
```
scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)
```
### Training and testing model
```
X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)
print("X_train\n")
print(X_train)
print("\nLenght of X_train ",len(X_train))
print("\nX_test\n")
print(X_test)
print("\nLenght of X_test ",len(X_test))
```
### 
## OUTPUT:
### Reading Data
<img width="631" height="563" alt="image" src="https://github.com/user-attachments/assets/a698f2d6-42a3-4e43-8147-826b684dc933" />

### X features & Y features
<img width="333" height="182" alt="image" src="https://github.com/user-attachments/assets/f5fcc8ce-eaa8-43c9-8a71-6f4f0e70e9fb" />

### Outliers
<img width="962" height="247" alt="image" src="https://github.com/user-attachments/assets/1c2cfba5-2d3c-4630-943d-12402b20311d" />

### Missing Values
<img width="577" height="586" alt="image" src="https://github.com/user-attachments/assets/d9f0a404-4d1b-45ea-b09a-f68e61edc35e" />

### Duplicates Identification
<img width="448" height="373" alt="image" src="https://github.com/user-attachments/assets/94adfd4b-2b73-4eef-b935-b8851fa51c3b" />

### Normalize the dataset
<img width="528" height="472" alt="image" src="https://github.com/user-attachments/assets/1985b927-521e-4661-a437-645641e9f46a" />

### SPLITTING THE TRAINING AND TESTING DATASET
<img width="567" height="417" alt="image" src="https://github.com/user-attachments/assets/f1eb69ed-07f4-444d-9fab-1c2873ff1d7e" />


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


