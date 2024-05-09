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
import pandas as pd
data=pd.read_csv("C:/Users/admin/Downloads/spam.csv",encoding = 'Windows-1252')
from sklearn.model_selection import train_test_split
```
```py
data
```
## Output:
![Screenshot 2024-05-09 192343](https://github.com/Mkumar262006/Implementation-of-SVM-For-Spam-Mail-Detection/assets/147139472/8e93fa1c-4e99-447c-9f2c-cd81193a6511)

```py
data.shape
```
## Output:
![Screenshot 2024-05-09 192435](https://github.com/Mkumar262006/Implementation-of-SVM-For-Spam-Mail-Detection/assets/147139472/9a49fc58-508b-4200-9beb-2a07e866c295)

```py
x=data["v2"].values
```
```py
y=data["v1"].values
```
```py
x.shape
```
## Output:
![Screenshot 2024-05-09 192613](https://github.com/Mkumar262006/Implementation-of-SVM-For-Spam-Mail-Detection/assets/147139472/e894b15e-50ed-4367-afc5-625fae5d3e01)
```py
y.shape
```
## Output:
![Screenshot 2024-05-09 192707](https://github.com/Mkumar262006/Implementation-of-SVM-For-Spam-Mail-Detection/assets/147139472/c6406dba-9140-4eb1-839b-5afd7f965457)
```py
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
```
```py
x_train
```
## Output:
![Screenshot 2024-05-09 192906](https://github.com/Mkumar262006/Implementation-of-SVM-For-Spam-Mail-Detection/assets/147139472/9056bda4-152c-454c-ad1a-60862f81b0d0)
```py
x_train.shape
```
## Output:
![Screenshot 2024-05-09 193019](https://github.com/Mkumar262006/Implementation-of-SVM-For-Spam-Mail-Detection/assets/147139472/d1a4b28b-c570-4ade-8780-91691e77fa8c)

```py
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
```
```py
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

```
```py
from sklearn.svm import SVC
```
```py
svc=SVC()
```
```py
svc.fit(x_train,y_train)
```
## Output:
![Screenshot 2024-05-09 193308](https://github.com/Mkumar262006/Implementation-of-SVM-For-Spam-Mail-Detection/assets/147139472/e2f38da3-5cfd-41d8-9422-017eb1289912)
```py
y_pred=svc.predict(x_test)
y_pred
```
## Output:
![Screenshot 2024-05-09 193337](https://github.com/Mkumar262006/Implementation-of-SVM-For-Spam-Mail-Detection/assets/147139472/d0e87433-07e3-4dcd-b842-3d2aa9d97a09)

```py
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
```
```py
acc=accuracy_score(y_test,y_pred)
acc
```
## Output:
![Screenshot 2024-05-09 193527](https://github.com/Mkumar262006/Implementation-of-SVM-For-Spam-Mail-Detection/assets/147139472/962cc45e-b60b-46e1-968b-d0965fce3086)
```py
con=confusion_matrix(y_test,y_pred)
print(con)
```
## Output:
![Screenshot 2024-05-09 193700](https://github.com/Mkumar262006/Implementation-of-SVM-For-Spam-Mail-Detection/assets/147139472/52a05bc0-aa29-4116-8430-ba0bf8d44367)
```py
cl=classification_report(y_test,y_pred)
print(cl)
```
## Output:
![Screenshot 2024-05-09 193803](https://github.com/Mkumar262006/Implementation-of-SVM-For-Spam-Mail-Detection/assets/147139472/cde9389a-4baf-4488-b61f-a7c0e3220ff4)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
