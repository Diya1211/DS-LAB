import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import confusion_matrix

df = pd.read_csv ('Iris.csv')

X = df.drop ('Species',axis = 1)

y = df['Species']

print (X)

print (y)

X_train,X_test,y_train,y_test = train_test_split (

X,y,test_size = 0.2,random_state = 42

)

classifier = GaussianNB ()

classifier.fit (X_train,y_train)

y_pred = classifier.predict (X_test)

confusion_mat = confusion_matrix (y_test,y_pred)

print ("Confusion Matrix : ")

print (confusion_mat)

tn = confusion_mat[0,0 ]

fp = confusion_mat[0,1 ]

fn = confusion_mat[1,0 ]

tp = confusion_mat[1,1 ]

accuracy = (tp +tn)/(tp +tn +fp +fn)

error_rate = 1 -accuracy

precision = tp /(tp +fp)

recall = tp /(tp +fn)

print ("Accuracy:",accuracy)

print ("Error Rate:",error_rate)

print ("Precision:",precision)

print ("Recall:",recall)
