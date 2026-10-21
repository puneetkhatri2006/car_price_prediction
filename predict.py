import pandas as pd
import joblib

#Load the trained model
model = joblib.load("model/car_price_model.pkl")

#Load the dataset to get the original feature columns
df = pd.read_csv("dataset/car data.csv")

#Remove dulpicate rows
df = df.drop_duplicates()

#Prepare features exactly as during training
x = df.drop(["Selling_Price","Car_Name"], axis=1)
x = pd.get_dummies(x, drop_first=True, dtype=int)

#Get the feature names used during training
feature_columns = x.columns

print("\nCar Price Prediction System")

#Get car details from the user
year = int(input("Enter manufacturing year:"))
present_price = float(input("Enter current showroom price in lakhs:"))
kms_driven = int(input("Enter kilometers driven:"))
fuel_type = input("Enter fuel type (Petrol/Diesel/CNG):")
seller_type = input("Enter seller type (Dealer/Individual):")
transmission = input("Enter transmission (Manual/Automatic):")
owner = int(input("Enter previous owners(0,1,2 or 3):"))

#Create input data
input_data = pd.DataFrame([{
    "Year": year,
    "Present_Price": present_price,
    "Kms_Driven": kms_driven,
    "Fuel_Type": fuel_type,
    "Seller_Type": seller_type,
    "Transmission": transmission,
    "Owner": owner
}])

#Encode categorical features
input_data = pd.get_dummies(input_data, drop_first=True, dtype=int)

#Match training columns
input_data = input_data.reindex(
    columns=feature_columns,
    fill_value=0
)

#Predict the price
predicted_price = model.predict(input_data)[0]

print(f"\nEstimated Selling Price: Rs. {predicted_price:.2f} lakhs")