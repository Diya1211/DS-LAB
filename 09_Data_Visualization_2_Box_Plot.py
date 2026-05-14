import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

titanic_df = sns.load_dataset ('titanic')

data = titanic_df[['sex','age','survived']]

data = data.dropna (subset = ['age'])

sns.boxplot (

x = 'sex',

y = 'age',

hue = 'survived',

data = data

)

plt.title ('Distribution of Age with respect to Gender and Survival')

plt.xlabel ('Gender')

plt.ylabel ('Age')

plt.show ()
