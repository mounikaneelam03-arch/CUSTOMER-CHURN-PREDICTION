from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load the model and dataset globals
model = None
median_monthly_charges = 70.35 # Default fallback

def load_resources():
    global model, median_monthly_charges
    try:
        with open("churn_model.pkl", "rb") as f:
            model = pickle.load(f)
    except FileNotFoundError:
        print("Model file 'churn_model.pkl' not found.")
        return False

    try:
        data_source = pd.read_csv("Telco_Cusomer_Churn.csv")
        data_source['TotalCharges'] = pd.to_numeric(data_source['TotalCharges'], errors='coerce')
        median_monthly_charges = data_source['MonthlyCharges'].median()
    except FileNotFoundError:
        print("Dataset not found. Using default median.")
    
    return True

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', prediction_text="Error: Model not loaded.", prediction_class="churn-yes")

    # Get form data
    input_data = {
        'gender': request.form['gender'],
        'SeniorCitizen': int(request.form['SeniorCitizen']),
        'Partner': request.form['Partner'],
        'Dependents': request.form['Dependents'],
        'tenure': int(request.form['tenure']),
        'PhoneService': request.form['PhoneService'],
        'MultipleLines': request.form['MultipleLines'],
        'InternetService': request.form['InternetService'],
        'OnlineSecurity': request.form['OnlineSecurity'],
        'OnlineBackup': request.form['OnlineBackup'],
        'DeviceProtection': request.form['DeviceProtection'],
        'TechSupport': request.form['TechSupport'],
        'StreamingTV': request.form['StreamingTV'],
        'StreamingMovies': request.form['StreamingMovies'],
        'Contract': request.form['Contract'],
        'PaperlessBilling': request.form['PaperlessBilling'],
        'PaymentMethod': request.form['PaymentMethod'],
        'MonthlyCharges': float(request.form['MonthlyCharges']),
        'TotalCharges': float(request.form['TotalCharges'])
    }

    input_df = pd.DataFrame(input_data, index=[0])

    # Feature Engineering
    # 1. Tenure Group
    bins = [0, 12, 24, 48, 60, 100]
    labels = ["0-12", "12-24", "24-48", "48-60", "60+"]
    input_df['TenureGroup'] = pd.cut(input_df['tenure'], bins=bins, labels=labels, right=False)

    # 2. AvgMonthlySpend
    input_df['AvgMonthlySpend'] = input_df['TotalCharges'] / (input_df['tenure'] + 1)

    # 3. HighCharges
    input_df['HighCharges'] = np.where(input_df['MonthlyCharges'] > median_monthly_charges, 1, 0)

    # One-Hot Encoding Construction
    try:
        model_features = model.feature_names_in_
        final_df = pd.DataFrame(0, index=[0], columns=model_features)
    except AttributeError:
        # Fallback if feature_names_in_ is missing, assuming standard structure
        # (This is risky but cleaner than crashing)
        return render_template('index.html', prediction_text="Error: Incompatible model version.", prediction_class="churn-yes")

    # Numeric features
    final_df['SeniorCitizen'] = input_df['SeniorCitizen']
    final_df['tenure'] = input_df['tenure']
    final_df['MonthlyCharges'] = input_df['MonthlyCharges']
    final_df['TotalCharges'] = input_df['TotalCharges']
    final_df['AvgMonthlySpend'] = input_df['AvgMonthlySpend']
    final_df['HighCharges'] = input_df['HighCharges']
    
    # Categorical mappings
    if input_df['gender'].iloc[0] == 'Male': final_df['gender_Male'] = 1
    if input_df['Partner'].iloc[0] == 'Yes': final_df['Partner_Yes'] = 1
    if input_df['Dependents'].iloc[0] == 'Yes': final_df['Dependents_Yes'] = 1
    if input_df['PhoneService'].iloc[0] == 'Yes': final_df['PhoneService_Yes'] = 1
    
    if input_df['MultipleLines'].iloc[0] == 'No phone service': final_df['MultipleLines_No phone service'] = 1
    elif input_df['MultipleLines'].iloc[0] == 'Yes': final_df['MultipleLines_Yes'] = 1
        
    if input_df['InternetService'].iloc[0] == 'Fiber optic': final_df['InternetService_Fiber optic'] = 1
    elif input_df['InternetService'].iloc[0] == 'No': final_df['InternetService_No'] = 1
        
    if input_df['OnlineSecurity'].iloc[0] == 'No internet service': final_df['OnlineSecurity_No internet service'] = 1
    elif input_df['OnlineSecurity'].iloc[0] == 'Yes': final_df['OnlineSecurity_Yes'] = 1
        
    if input_df['OnlineBackup'].iloc[0] == 'No internet service': final_df['OnlineBackup_No internet service'] = 1
    elif input_df['OnlineBackup'].iloc[0] == 'Yes': final_df['OnlineBackup_Yes'] = 1

    if input_df['DeviceProtection'].iloc[0] == 'No internet service': final_df['DeviceProtection_No internet service'] = 1
    elif input_df['DeviceProtection'].iloc[0] == 'Yes': final_df['DeviceProtection_Yes'] = 1

    if input_df['TechSupport'].iloc[0] == 'No internet service': final_df['TechSupport_No internet service'] = 1
    elif input_df['TechSupport'].iloc[0] == 'Yes': final_df['TechSupport_Yes'] = 1

    if input_df['StreamingTV'].iloc[0] == 'No internet service': final_df['StreamingTV_No internet service'] = 1
    elif input_df['StreamingTV'].iloc[0] == 'Yes': final_df['StreamingTV_Yes'] = 1

    if input_df['StreamingMovies'].iloc[0] == 'No internet service': final_df['StreamingMovies_No internet service'] = 1
    elif input_df['StreamingMovies'].iloc[0] == 'Yes': final_df['StreamingMovies_Yes'] = 1

    if input_df['Contract'].iloc[0] == 'One year': final_df['Contract_One year'] = 1
    elif input_df['Contract'].iloc[0] == 'Two year': final_df['Contract_Two year'] = 1

    if input_df['PaperlessBilling'].iloc[0] == 'Yes': final_df['PaperlessBilling_Yes'] = 1

    if input_df['PaymentMethod'].iloc[0] == 'Credit card (automatic)': final_df['PaymentMethod_Credit card (automatic)'] = 1
    elif input_df['PaymentMethod'].iloc[0] == 'Electronic check': final_df['PaymentMethod_Electronic check'] = 1
    elif input_df['PaymentMethod'].iloc[0] == 'Mailed check': final_df['PaymentMethod_Mailed check'] = 1

    tg = input_df['TenureGroup'].iloc[0]
    if tg == '12-24': final_df['TenureGroup_12-24'] = 1
    elif tg == '24-48': final_df['TenureGroup_24-48'] = 1
    elif tg == '48-60': final_df['TenureGroup_48-60'] = 1
    elif tg == '60+': final_df['TenureGroup_60+'] = 1

    # Predict
    prediction = model.predict(final_df)
    prediction_proba = model.predict_proba(final_df)
    
    result = "Likely to Churn" if prediction[0] == 1 else "Unlikely to Churn"
    confidence = prediction_proba[0][1] if prediction[0] == 1 else prediction_proba[0][0]
    css_class = "churn-yes" if prediction[0] == 1 else "churn-no"
    
    prediction_text = f"Prediction: {result} (Confidence: {confidence:.2f})"

    return render_template('index.html', prediction_text=prediction_text, prediction_class=css_class)

if __name__ == '__main__':
    load_resources()
    app.run(debug=True, port=5000)
