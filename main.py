import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load the dataset
df = pd.read_csv("dataset/car data.csv")

# Display the first five rows
print(df.head())

# Display the dataset dimensions
print(df.shape)

# Display column information
df.info()

# Check missing values
print(df.isnull().sum())

#Display all column names
print("\nColumn Names:")
print(df.columns)

#Display statistical information
print("\nStatistical Summary:")
print(df.describe())

#Check duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

#Remove duplicate rows
df=df.drop_duplicates()

#Display the dataset size after removing duplicates
print("\nDataset shape after removing duplicates:")
print(df.shape)

#Check duplicates again
print("\nDuplicate rows after cleaning:")
print(df.duplicated().sum())

#Graph 1: Car year Vs selling price
plt.figure(figsize=(8,5))

sns.scatterplot(
    data=df,
    x="Year",
    y="Selling_Price"
)

plt.title("Car Manufacturing Year vs Selling Price")
plt.xlabel("Manufacturing Year")
plt.ylabel("Selling Price (in lakhs)")
plt.tight_layout()

plt.savefig("graphs/year_vs_price.png")
plt.show()

#Graph 2: Kilometers driven vs Selling Price
plt.figure(figsize=(8,5))

sns.scatterplot(
    data=df,
    x="Kms_Driven",
    y="Selling_Price"
)

plt.title("Kilometers Driven vs Selling Price")
plt.xlabel("Kilometers Driven")
plt.ylabel("Selling Price(in lakhs)")

plt.tight_layout()

plt.savefig("graphs/km_driven_vs_price.png")
plt.show()

#Select features and target
x = df.drop(["Selling_Price","Car_Name"], axis=1)
y = df["Selling_Price"]

print("\nFeatures (x):")
print(x.head())

print("\nTarget (y):")
print(y.head())

print("\nFeature shape:", x.shape)
print("Target shape:", y.shape)

#Encode categorical features
x = pd.get_dummies(x,drop_first=True, dtype=int)

print("\nEncoded Features:")
print(x.head())

print("\nEncoded Feature shape:")
print(x.shape)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42
)

print("\nTraining and Testing Data")
print("x_train shape:",x_train.shape)
print("x_test shape:",x_test.shape)
print("y_train shape:",y_train.shape)
print("y_test shape:",y_test.shape)

# Train Linear Regression Model
# create the model
model = LinearRegression()

# Train the model
model.fit(x_train, y_train)

print("\nLinear Regression model trained successfully!")

# Evaluate the model
# Predict prices for testing data
y_pred =model.predict(x_test)

# Calculate evaluation metrices
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation")
print("Mean Absolute Error:",mae)
print("Mean Squared Erroe:",mse)
print("R2 Score:",r2)

# Create the model
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train the model
rf_model.fit(x_train, y_train)

# Make predictions
rf_pred = rf_model.predict(x_test)

# Evaluate the model
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_mse = mean_squared_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

print("\nRandom Forest Results")
print("MAE:", rf_mae)
print("MSE:", rf_mse)
print("R2 Score:",rf_r2)

# Save the trained linear Regression model
joblib.dump(model, "model/car_price_model.pkl")

print("\nBest model saved successfully!")

# Actual vs Predicted Price graph
plt.figure(figsize=(8,5))

plt.scatter(y_test, y_pred)

# Perfect prediction refrences line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--",
)

plt.xlabel("Actual Selling Price (in lakhs)")
plt.ylabel("Predicted Selling Price (in lakhs)")
plt.title("Actual vs Predicted Selling Price")

plt.tight_layout()
plt.savefig("graphs/actual_vs_predicted.png")
plt.show()

# Feature Importance Visualization
importance = pd.DataFrame({
    "Feature": x_train.columns,
    "Importance": rf_model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance)

plt.figure(figsize=(10,6))

plt.barh(
    importance["Feature"],
    importance["Importance"]
)

plt.xlabel("Important Score")
plt.ylabel("Car Features")
plt.title("Feature Importance in Car Price Prediction")

plt.gca().invert_yaxis()
plt.tight_layout()

plt.savefig("graphs/feature_importance.png")
plt.show()

# Compare Linear Regression and Random Forest
model_names = ["Linear Regression", "Random Forest"]

mae_values = [mae, rf_mae]
r2_values = [r2, rf_r2]

plt.figure(figsize=(8,5))
plt.bar(model_names, mae_values)

plt.title("Model Comparison - MAE")
plt.xlabel("Machine Learning Models")
plt.ylabel("Mean Absolute Error")

plt.tight_layout()
plt.savefig("graphs/model_comparison_mae.png")
plt.show()

plt.figure(figsize=(8,5))
plt.bar(model_names, r2_values)

plt.title("Model Comparison - R2 Score")
plt.xlabel("Machine Learning Models")
plt.ylabel("R2 Score")

plt.tight_layout()
plt.savefig("graphs/model_comparison_r2.png")
plt.show()