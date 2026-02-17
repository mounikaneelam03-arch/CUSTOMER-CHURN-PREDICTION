# Customer Churn Prediction (Flask Web App)

This project has been migrated from a Streamlit application to a Flask web application.

## How to Run

1.  Stop the previous Streamlit process if it is still running (Ctrl+C).
2.  Ensure you have Flask installed:
    ```bash
    pip install flask pandas scikit-learn
    ```
3.  Run the application:
    ```bash
    python app.py
    ```
4.  Open your browser and navigate to:
    [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Features

-   **Web UI**: A clean, responsive interface built with Bootstrap.
-   **Prediction**: Input customer details and get real-time churn predictions.
-   **Backend**: Flask handles model loading and data processing.
