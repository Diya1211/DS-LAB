import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

# Load dataset
df = pd.read_csv("Iris.csv")

# Display first row
print("First 5 rows:")
print(df.head())

# Display the last row 
print("\nLast 5 rows:")
print(df.tail())

# Shape of dataset(rows , cols)
print("\nShape of dataset:")
print(df.shape)

# Column names
print("\nColumns:")
print(df.columns)

# Datatype of cols
print("\nData types:")
print(df.dtypes)

# Provides statistics
print("\nDescriptive statistics:")
print(df.describe())

# Check missing value true=missing
print("\nNull values check:")
print(df.isnull())

# Counts missing values column-wise 0=no missing
print("\nNull values sum per column:")
print(df.isnull().sum())

# Total Missing Values
print("\nTotal missing values:")
print(df.isnull().sum().sum())

# Displays only Species column.
print("\nSpecies column:")
print(df['Species'])

# Display a specific row
print("\nRow at index 5:")
print(df.iloc[5])

# Convert datatype
df['PetalLengthCm'] = df['PetalLengthCm'].astype(int)

# Normalization
scaler = MinMaxScaler()

df[['SepalLengthCm',
    'SepalWidthCm',
    'PetalLengthCm',
    'PetalWidthCm']] = scaler.fit_transform(
    df[['SepalLengthCm',
        'SepalWidthCm',
        'PetalLengthCm',
        'PetalWidthCm']]
)

# Label Encoding
label_encoder = preprocessing.LabelEncoder()

df['Species'] = label_encoder.fit_transform(df['Species'])

# Final output
print("\nFinal transformed dataframe (head):")
print(df.head())
