#Importing libraries
import pandas as pd
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

digits = load_digits()
dir(digits)
#Importing the datsets
df = pd.DataFrame(digits.data)
#Appending last column
df['target'] = digits.target
X = df.drop('target',axis='columns')
y = df.target

#Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

#Fitting the model 
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=20)
model.fit(X_train, y_train)

#Testing the model
model.score(X_test, y_test)

#Predicting the model
y_predicted = model.predict(X_test)

#Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)
cm
#PLotting the confusion matrix
import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')