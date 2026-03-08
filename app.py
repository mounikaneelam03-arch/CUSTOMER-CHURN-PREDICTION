from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)
load_resources()

# Globals
model = None
median_monthly_charges = 70.35

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_resources():
    global model, median_monthly_charges

    try:
        model_path = os.path.join(BASE_DIR, "churn_model.pkl")
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        print("Model loaded successfully")

    except Exception as e:
        print("Error loading model:", e)
        return False

    try:
        data_path = os.path.join(BASE_DIR, "Telco_Cusomer_Churn.csv")
        data_source = pd.read_csv(data_path)

        data_source['TotalCharges'] = pd.to_numeric(
            data_source['TotalCharges'], errors='coerce'
        )

        median_monthly_charges = data_source['MonthlyCharges'].median()

    except Exception as e:
        print("Dataset load failed, using default median:", e)

    return True


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    if model is None:
        return render_template(
            'index.html',
            prediction_text="Error: Model not loaded.",
            prediction_class="churn-yes"
        )

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
    bins = [0, 12, 24, 48, 60, 100]
    labels = ["0-12", "12-24", "24-48", "48-60", "60+"]

    input_df['TenureGroup'] = pd.cut(
        input_df['tenure'], bins=bins, labels=labels, right=False
    )

    input_df['AvgMonthlySpend'] = (
        input_df['TotalCharges'] / (input_df['tenure'] + 1)
    )

    input_df['HighCharges'] = np.where(
        input_df['MonthlyCharges'] > median_monthly_charges, 1, 0
    )

    try:
        model_features = model.feature_names_in_
        final_df = pd.DataFrame(0, index=[0], columns=model_features)

    except AttributeError:
        return render_template(
            'index.html',
            prediction_text="Error: Model incompatible.",
            prediction_class="churn-yes"
        )

    # Numeric Features
    final_df['SeniorCitizen'] = input_df['SeniorCitizen']
    final_df['tenure'] = input_df['tenure']
    final_df['MonthlyCharges'] = input_df['MonthlyCharges']
    final_df['TotalCharges'] = input_df['TotalCharges']
    final_df['AvgMonthlySpend'] = input_df['AvgMonthlySpend']
    final_df['HighCharges'] = input_df['HighCharges']

    # Categorical Features
    if input_df['gender'].iloc[0] == 'Male':
        final_df['gender_Male'] = 1

    if input_df['Partner'].iloc[0] == 'Yes':
        final_df['Partner_Yes'] = 1

    if input_df['Dependents'].iloc[0] == 'Yes':
        final_df['Dependents_Yes'] = 1

    if input_df['PhoneService'].iloc[0] == 'Yes':
        final_df['PhoneService_Yes'] = 1

    if input_df['Contract'].iloc[0] == 'One year':
        final_df['Contract_One year'] = 1
    elif input_df['Contract'].iloc[0] == 'Two year':
        final_df['Contract_Two year'] = 1

    if input_df['PaperlessBilling'].iloc[0] == 'Yes':
        final_df['PaperlessBilling_Yes'] = 1

    tg = input_df['TenureGroup'].iloc[0]

    if tg == '12-24':
        final_df['TenureGroup_12-24'] = 1
    elif tg == '24-48':
        final_df['TenureGroup_24-48'] = 1
    elif tg == '48-60':
        final_df['TenureGroup_48-60'] = 1
    elif tg == '60+':
        final_df['TenureGroup_60+'] = 1

    prediction = model.predict(final_df)
    prediction_proba = model.predict_proba(final_df)

    result = "Likely to Churn" if prediction[0] == 1 else "Unlikely to Churn"

    confidence = (
        prediction_proba[0][1]
        if prediction[0] == 1
        else prediction_proba[0][0]
    )

    css_class = "churn-yes" if prediction[0] == 1 else "churn-no"

    prediction_text = f"Prediction: {result} (Confidence: {confidence:.2f})"

    return render_template(
        'index.html',
        prediction_text=prediction_text,
        prediction_class=css_class
    )


if __name__ == '__main__':
    

    port = int(os.environ.get("PORT", 5000))

    app.run(host='0.0.0.0', port=port)
