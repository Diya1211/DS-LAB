import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from scipy import stats

# Load dataset
df = pd.read_csv("Student_performance_data _.csv")

# Display first 5 rows
print("Head:")
print(df.head())

# Shape
print("\nShape:")
print(df.shape)

# Columns
print("\nColumns:")
print(df.columns)

# Data types
print("\nData types:")
print(df.dtypes)

# Statistics
print("\nDescribe:")
print(df.describe())

# Missing values
print("\nNull values sum:")
print(df.isnull().sum())
print("\nTotal null values:")
print(df.isnull().sum().sum())

# Fill missing values with mean
for column in df.select_dtypes(include=['int64', 'float64']).columns:
    df[column] = df[column].fillna(df[column].mean())

print("\nNull values after filling:")
print(df.isnull().sum())

# Label Encoding for object columns
label_encoder = LabelEncoder()
for column in df.select_dtypes(include='object').columns:
    df[column] = label_encoder.fit_transform(df[column].astype(str))

# Numeric columns for outlier detection
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Boxplot for outlier detection
plt.figure(figsize=(12,6))
df[numeric_columns].boxplot()
plt.title("Boxplot for Outlier Detection")
plt.xticks(rotation=45)
plt.show()

# Detect outliers using Z-score
z = np.abs(stats.zscore(df[numeric_columns]))
print("\nOutlier indices:")
print(np.where(z > 3))

# Filter outliers
new_df = df[(z < 3).all(axis=1)]

# Histogram before transformation
numeric_col = numeric_columns[0]
new_df[numeric_col].plot(kind='hist')
plt.title("Histogram Before Transformation")
plt.show()

# Log transformation to reduce skewness
new_df['Log_Transform'] = np.log10(new_df[numeric_col] + 1)

# Histogram after transformation
new_df['Log_Transform'].plot(kind='hist')
plt.title("Histogram After Log Transformation")
plt.show()
