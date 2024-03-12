# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: V.Gajalakshmi
RegisterNumber: 212223040047

/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Hema Dharshini N
RegisterNumber: 212223220034 
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse) 
*/
```

## Output:

![Screenshot 2024-03-04 174024](https://github.com/Gajalakshmivelmurugan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144871940/ef31fa59-6293-4402-90f0-6d61c7de8529)
![Screenshot 2024-03-04 174036](https://github.com/Gajalakshmivelmurugan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144871940/c2e8cfb3-e82f-4f96-99a4-7643e0fbd051)
![Screenshot 2024-03-04 174056](https://github.com/Gajalakshmivelmurugan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144871940/55ecc405-b10b-4ae0-a809-8537eeb11fa9)
![Screenshot 2024-03-04 174110](https://github.com/Gajalakshmivelmurugan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144871940/3815d668-7933-48e8-99b6-510051997135)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
