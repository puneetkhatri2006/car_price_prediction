Car Price Prediction 🚗

##📌 Project Overview

Car Price Prediction is a Machine Learning project that predicts the estimated selling price of a used car based on its features.

The project uses historical car data to train a Linear Regression model and provides a user-friendly web interface built with Streamlit.

##🎯 Objectives

- Analyze used-car data.
- Explore relationships between car features and selling prices.
- Train and evaluate a Machine Learning model.
- Predict selling prices based on user inputs.
- Display predictions through an interactive web application.

##🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit

##📂 Project Structure

Car_Price_Prediction/
├── dataset/
│   └── car data.csv
├── graphs/
│   ├── year_vs_price.png
│   ├── km_driven_vs_price.png
│   ├── actual_vs_predicted.png
│   ├── feature_importance.png
│   ├── model_comparison_mae.png
│   └── model_comparison_r2.png
├── model/
│   └── car_price_model.pkl
├── main.py
├── predict.py
├── app.py
├── requirements.txt
└── README.md

##📊 Dataset

The dataset contains information about used cars, including:

- Manufacturing year
- Showroom price
- Kilometers driven
- Fuel type
- Seller type
- Transmission type
- Previous owners
- Selling price

##🤖 Machine Learning

The project explores Linear Regression and Random Forest Regression.
The Linear Regression model is used for price prediction.
Model performance is evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R² Score

##🌐 Web Application

The Streamlit application allows users to:

1. Enter car details.
2. Click the Predict Car Price button.
3. View the estimated selling price.
4. Compare the showroom price with the predicted selling price.

##▶️ How to Run the Project

Step 1: Install dependencies
Open the terminal in the project folder and run:
pip install -r requirements.txt

Step 2: Run the web application
streamlit run app.py
The application will open in your web browser.

##📈 Key Features

- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Data visualization
- Machine Learning model training
- Model evaluation
- Interactive price prediction

##🎓 Project Purpose

This project was developed as a beginner-level Machine Learning project to gain practical experience in data analysis, model training, evaluation, and web application development.

