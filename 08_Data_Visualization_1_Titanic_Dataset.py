import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

df = pd.read_csv ('Titanic.csv')

print ("Shape of Dataset:")

print (df.shape)

print ("\nFirst 5 Rows:")

print (df.head ())

print ("\nLast 5 Rows:")

print (df.tail ())

print ("\nDataset Info:")

print (df.info ())

print ("\nMissing Values:")

print (df.isnull ().sum ())

df.drop (['Cabin'],axis = 1,inplace = True)

print ("\nEmbarked Value Counts:")

print (df["Embarked"].value_counts ())

df["Embarked"].fillna ("S",inplace = True)

df['Age'].fillna (df['Age'].mean (),inplace = True)

print ("\nMissing Values After Cleaning:")

print (df.isnull ().sum ())

print ("\nStatistical Summary:")

print (df.describe ())

print ("\nSurvival Count:")

print (df["Survived"].value_counts ())

sns.countplot (x = "Survived",data = df)

plt.title ("Survival Count")

plt.show ()

sns.countplot (x = "Sex",hue = "Survived",data = df)

plt.title ("Gender vs Survival")

plt.show ()

sns.countplot (x = "Pclass",hue = "Survived",data = df)

plt.title ("Passenger Class vs Survival")

plt.show ()

sns.countplot (x = "Embarked",hue = "Survived",data = df)

plt.title ("Embarked vs Survival")

plt.show ()

sns.countplot (x = "SibSp",hue = "Survived",data = df)

plt.title ("Siblings/Spouses vs Survival")

plt.show ()
