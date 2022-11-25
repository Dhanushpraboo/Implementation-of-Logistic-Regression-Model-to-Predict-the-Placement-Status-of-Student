# Implementation of Logistic Regression Model to Predict the Placement Status of Student
### AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

### Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook
## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3. Import LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5. Predict the values of array.
6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7. Apply new unknown values.

### Program:
~~~
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: S Dhanush Praboo
RegisterNumber:  212221230019
*/
import pandas as pd
data = pd.read_csv("Placement_Data.csv")
data.head()

data1 = data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x = data1.iloc[:,:-1]
x

y = data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)

y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
~~~

### Output:
![image](https://user-images.githubusercontent.com/94426323/200584345-950f2bf4-cdfa-4ce9-a04d-abde691a8e77.png)
![image](https://user-images.githubusercontent.com/94426323/200584377-7e5e0e67-7464-446a-b663-bcc4c4b80114.png)
![image](https://user-images.githubusercontent.com/94426323/200584421-0f823124-7238-4fb9-8e08-772437f76e68.png)
![image](https://user-images.githubusercontent.com/94426323/200584460-6725701f-b5e6-4ac4-a0cb-c37e68c0c1f8.png)
![image](https://user-images.githubusercontent.com/94426323/200584485-d1879014-6330-4364-9418-adb354a6b824.png)
![image](https://user-images.githubusercontent.com/94426323/200584544-84d22128-43ad-4c6f-9cfd-8fdacb339fcf.png)
![image](https://user-images.githubusercontent.com/94426323/200584570-5537c9bd-650b-4282-8c9a-daecc9e2d7d9.png)
![image](https://user-images.githubusercontent.com/94426323/200584587-13f363cb-bda2-4e15-b79c-98e106286667.png)
![image](https://user-images.githubusercontent.com/94426323/200584622-7697e63e-d268-4197-9c44-57d42d938cd8.png)
![image](https://user-images.githubusercontent.com/94426323/200584640-2ccb10ed-9b73-4f76-ab0b-6cb99d7eab77.png)
![image](https://user-images.githubusercontent.com/94426323/200584674-41ed6ac0-8037-431c-a3c7-bb0e133a21a0.png)
![image](https://user-images.githubusercontent.com/94426323/200584698-c4862f40-f745-4ae2-8bad-ac97e575d9a8.png)
![image](https://user-images.githubusercontent.com/94426323/200584729-9542c981-bf09-48ce-b361-9f9f1faa7331.png)



### Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
