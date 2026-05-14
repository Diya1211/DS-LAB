# STEP 1 : Import Libraries
import pandas as pd
import numpy as np

# STEP 2 : Load Dataset
df = pd.read_csv("Iris.csv")

# STEP 3 : Display Dataset
print("FIRST 5 ROWS")
print(df.head())

# STEP 4 : Shape of Dataset
print("\nSHAPE OF DATASET")
print(df.shape)

# STEP 5 : Column Names
print("\nCOLUMN NAMES")
print(df.columns)

# STEP 6 : Data Types
print("\nDATA TYPES")
print(df.dtypes)

# STEP 7 : Mean
print("\nMEAN VALUES")
print("SepalLengthCm Mean:", df["SepalLengthCm"].mean())
print("SepalWidthCm Mean:", df["SepalWidthCm"].mean())

# STEP 8 : Median
print("\nMEDIAN VALUES")
print("PetalLengthCm Median:", df["PetalLengthCm"].median())
print("PetalWidthCm Median:", df["PetalWidthCm"].median())

# STEP 9 : Mode
print("\nMODE VALUES")
print(df["SepalLengthCm"].mode())

# STEP 10 : Standard Deviation
print("\nSTANDARD DEVIATION")
print(df["SepalLengthCm"].std())

# STEP 11 : Minimum and Maximum
print("\nMINIMUM VALUES")
print(df.min(numeric_only=True))
print("\nMAXIMUM VALUES")
print(df.max(numeric_only=True))

# STEP 12 : Group By Species
print("\nGROUPED STATISTICS")
grouped = df.groupby("Species").mean()
print(grouped)

# STEP 13 : Create Lists for Each Species
setosa = df[df["Species"] == "Iris-setosa"]["SepalLengthCm"].tolist()
versicolor = df[df["Species"] == "Iris-versicolor"]["SepalLengthCm"].tolist()
virginica = df[df["Species"] == "Iris-virginica"]["SepalLengthCm"].tolist()

print("\nSETOSA LIST")
print(setosa)
print("\nVERSICOLOR LIST")
print(versicolor)
print("\nVIRGINICA LIST")
print(virginica)

# STEP 14 : Describe Function
print("\nFULL DATASET STATISTICS")
print(df.describe())

# STEP 15 : Statistics of Iris-setosa
print("\nIRIS-SETOSA STATISTICS")
print(df[df["Species"] == "Iris-setosa"].describe())

# STEP 16 : Statistics of Iris-versicolor
print("\nIRIS-VERSICOLOR STATISTICS")
print(df[df["Species"] == "Iris-versicolor"].describe())

# STEP 17 : Statistics of Iris-virginica
print("\nIRIS-VIRGINICA STATISTICS")
print(df[df["Species"] == "Iris-virginica"].describe())
