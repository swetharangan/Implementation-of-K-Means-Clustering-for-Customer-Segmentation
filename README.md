# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1. Start

Step 2.Import Necessary Libraries and Load Data

Step 3.Split Dataset into Training and Testing Sets

Step 4.Train the Model Using Stochastic Gradient Descent (SGD)

Step 5.Make Predictions and Evaluate Accuracy

Step 6.Generate Confusion Matrix

Step 7. Stop

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: SWETHA R
RegisterNumber:  212223040227
*/
```
```py

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
print(df.head())

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sgd_clf = SGDClassifier(max_iter = 1000, tol = 1e-3)
sgd_clf.fit(X_train, y_train)

y_pred = sgd_clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

```
## Output:
![Screenshot 2024-10-19 132439](https://github.com/user-attachments/assets/85068793-799f-47fd-8df9-a346ac02b0f9)


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
