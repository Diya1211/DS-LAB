import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler,LabelEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score

df = pd.read_csv ("Social_Network_Ads.csv")

label_encoder = LabelEncoder ()

df["Gender"] = label_encoder.fit_transform (df["Gender"])

X = df[["Gender","Age","EstimatedSalary"]]

y = df["Purchased"]

X_train,X_test,y_train,y_test = train_test_split (

X,y,test_size = 0.2,random_state = 42

)

scaler = StandardScaler ()

X_train_scaled = scaler.fit_transform (X_train)

X_test_scaled = scaler.transform (X_test)

log_reg = LogisticRegression ()

log_reg.fit (X_train_scaled,y_train)

y_pred = log_reg.predict (X_test_scaled)

cm = confusion_matrix (y_test,y_pred)

TN,FP,FN,TP = cm.ravel ()

accuracy = accuracy_score (y_test,y_pred)

error_rate = 1 -accuracy

precision = precision_score (y_test,y_pred)

recall = recall_score (y_test,y_pred)

print ("Confusion Matrix:")

print (cm)

print (f"True Negatives (TN): {TN }")

print (f"False Positives (FP): {FP }")

print (f"False Negatives (FN): {FN }")

print (f"True Positives (TP): {TP }")

print (f"Accuracy: {accuracy :.4f}")

print (f"Error Rate: {error_rate :.4f}")

print (f"Precision: {precision :.4f}")

print (f"Recall: {recall :.4f}")
