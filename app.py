import streamlit as st
import pandas as pd
import joblib

#Load the trained model
model = joblib.load("model/car_price_model.pkl")

#Load model evaluation results
mae = 1.472892414003328
mse = 6.270752956826566
r2 = 0.751815421583164

#Load the dataset to get the training feature columns
df = pd.read_csv("dataset/car data.csv")
df = df.drop_duplicates()

x = df.drop(["Selling_Price","Car_Name"], axis=1)
x = pd.get_dummies(x, drop_first=True, dtype=int)

feature_columns = x.columns

#Page configuration
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="centered"
)

#App heading
st.title("Car Price Prediction System")
st.write("Enter your car details to estimate its selling price.")

#Input heading
col1, col2 = st.columns(2)

with col1:
    year = st.number_input(
        "Manufacturing Year",
        min_value=2000,
        max_value=2026,
        value=2020,
        step=1
    )
    
    present_price = st.number_input(
        "Present Showroom Price(Rs. lakhs)",
        min_value=0.1,
        value=5.0,
        step=0.5
    )
    
    kms_driven = st.number_input(
        "Kilometers Driven",
        min_value=0,
        value=25000,
        step=1000
    )
    
    fuel_type = st.selectbox(
        "Fuel Type",
        ["Petrol", "Diesel", "CNG"]
    )
    
with col2:
    seller_type = st.selectbox(
        "Seller Type",
        ["Dealer", "Individual"]
    )
    
    transmission = st.selectbox(
        "Transmission",
        ["Manual", "Automatic"]
    )
    
    owner = st.selectbox(
        "Previous Owner",
        [0, 1, 2, 3]
    )
    
#Prediction button
if st.button("Predict Car Price",type="primary"):
    
    #Prepare input data
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
    input_data = pd.get_dummies(
        input_data,
        drop_first=True,
        dtype=int
    )

    #Match the training feature columns
    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    #Predict
    predicted_price = model.predict(input_data)[0]

    #Display the result
    st.success(f"The estimated selling price of the car is: Rs. {predicted_price:.2f} lakhs")
    
    st.caption("This is a machine learning estimate, not a guaranteed market price.")
    
    #Compare showroom price and predicted selling price
    st.subheader("Price Comparison")
    
    comparison = pd.DataFrame({
        "Price Type": [
            "Showroom Price",
            "Predicted Selling Price"
        ],
        "Price (Lakhs)": [
            present_price,
            predicted_price
        ]
    })
    
    st.bar_chart(
        comparison.set_index("Price Type")
    )
    
    # Model Performance Section
    st.divider()
    
    st.subheader("Model Performance")
    col1, col2, col3 = st.columns(3)
    
    col1.metric("MAE", f"{mae:.2f}")
    col2.metric("MSE", f"{mse:.2f}")
    col3.metric("R2 Score", f"{r2:.2%}")
    
    st.caption(
        "These metric evaluate the model on a test dataset."
        "Actual prediction accuracy may vary.")

    