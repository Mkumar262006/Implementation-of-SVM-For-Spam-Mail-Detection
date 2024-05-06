# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
STEP 1 : Start
STEP 2 : Preprocessing the data
STEP 3 : Feature Extraction
STEP 4 : Training the SVM model
STEP 5 : Model Evalutaion
STEP 6 : Stop

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: MANOJ KUMAR S
RegisterNumber: 212223240082
*/
```
```py
import chardet
file="C:/Users/admin/Downloads/spam.csv"
with open(file, 'rb') as rawdata:
    result=chardet.detect(rawdata.read(100000))
result    
```
## Output:
![Screenshot 2024-05-06 134452](https://github.com/Mkumar262006/Implementation-of-SVM-For-Spam-Mail-Detection/assets/147139472/22d1e7a8-7af2-45ca-bc4e-9a3c30878e7f)
```py
import pandas as pd
data=pd.read_csv("C:/Users/admin/Downloads/spam.csv",encoding='Windows-1252')
```
```py
data.head()
```
## Output:
![Screenshot 2024-05-06 134814](https://github.com/Mkumar262006/Implementation-of-SVM-For-Spam-Mail-Detection/assets/147139472/597fb3ba-3a0d-4670-8487-b7b3cc8bc246)
```py
data.info()
```
## Output:
![Screenshot 2024-05-06 134939](https://github.com/Mkumar262006/Implementation-of-SVM-For-Spam-Mail-Detection/assets/147139472/77c00cfa-8a46-40c4-8bc4-314568fc6b90)
```py
data.isnull().sum()
```
## Output:
![Screenshot 2024-05-06 135107](https://github.com/Mkumar262006/Implementation-of-SVM-For-Spam-Mail-Detection/assets/147139472/8da17406-c058-4e34-afcc-74ea0be0e49d)
```py
x=data["v1"].values
```
```py
y=data["v2"].values
```
```py
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
```
```py
from sklearn.feature_extraction.text import CountVectorizer
#CountVectorizer is a method to convert text to numerical data.The text is tranaformed to a sparse matrix
cv=CountVectorizer()
```
```py
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

```
```py
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
```
## Output:
![Screenshot 2024-05-06 135452](https://github.com/Mkumar262006/Implementation-of-SVM-For-Spam-Mail-Detection/assets/147139472/84677888-cf80-47a0-be65-406850a6ca38)
```py
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
## Output:
![Screenshot 2024-05-06 135605](https://github.com/Mkumar262006/Implementation-of-SVM-For-Spam-Mail-Detection/assets/147139472/d7401f76-7cad-4621-9e42-7f52b02f2564)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
