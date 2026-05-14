# Step 1 : Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

# Step 2 : Load dataset
data = pd.read_csv("HousingData.csv")

# Step 3 : Display first 5 rows
print("Data head:")
print(data.head())

# Step 4 : Display information about dataset
print("\nData info:")
print(data.info())

# Step 5 : Check shape of dataset
print("\nShape:")
print(data.shape)

# Step 6 : Check column names
print("\nColumns:")
print(data.columns)

# Step 7 : Check missing values
print("\nNull values sum:")
print(data.isnull().sum())

# Step 8 : Separate independent and dependent variables
X = data.drop("MEDV", axis=1)
y = data["MEDV"]

# Step 9 : Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Step 10 : Handle missing values using mean
imputer = SimpleImputer(strategy='mean')

# Step 11 : Fit and transform training data
X_train_imputed = imputer.fit_transform(X_train)

# Step 12 : Transform testing data
X_test_imputed = imputer.transform(X_test)

# Step 13 : Convert arrays back to DataFrame
X_train_imputed = pd.DataFrame(
    X_train_imputed,
    columns=X_train.columns
)
X_test_imputed = pd.DataFrame(
    X_test_imputed,
    columns=X_test.columns
)

# Step 14 : Create Linear Regression model
model = LinearRegression()

# Step 15 : Train the model
model.fit(X_train_imputed, y_train)

# Step 16 : Make predictions
y_pred = model.predict(X_test_imputed)

# Step 17 : Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print("\nMean Squared Error :", mse)

# Step 18 : Calculate Root Mean Squared Error
rmse = np.sqrt(mse)
print("Root Mean Squared Error :", rmse)

# Step 19 : Calculate R2 Score
r2 = r2_score(y_test, y_pred)
print("R2 Score :", r2)

# Step 20 : Compare actual and predicted values
result = pd.DataFrame({
    "Actual Price": y_test,
    "Predicted Price": y_pred
})
print("\nActual vs Predicted (first 5):")
print(result.head())

# Step 21 : Display coefficients
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})
print("\nModel Coefficients:")
print(coefficients)

# Step 22 : Plot Actual vs Predicted values
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()
