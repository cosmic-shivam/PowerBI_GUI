from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import pickle
import os
app = Flask(__name__)

# Load trained model
with open('loan_approval_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load dataset
dataset = pd.read_csv('loan_approval_dataset.csv')

# Create mappings for education and self_employed
education_map = {1.0: 'Graduate', 0.0: 'Not Graduate'}
self_employed_map = {1.0: 'Yes', 0.0: 'No'}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_form')
def predict_form():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        form_data = request.form

        # Create a DataFrame from the input
        data = {
            'no_of_dependents': [int(form_data['no_of_dependents'])],
            'education': [float(form_data['education'])],
            'self_employed': [float(form_data['self_employed'])],
            'income_annum': [int(form_data['income_annum'])],
            'loan_amount': [int(form_data['loan_amount'])],
            'loan_term': [int(form_data['loan_term'])],
            'cibil_score': [int(form_data['cibil_score'])],
            'residential_assets_value': [int(form_data['residential_assets_value'])],
            'commercial_assets_value': [int(form_data['commercial_assets_value'])],
            'luxury_assets_value': [int(form_data['luxury_assets_value'])],
            'bank_asset_value': [int(form_data['bank_asset_value'])]
        }
        df_input = pd.DataFrame(data)

        # Convert DataFrame to a NumPy array without feature names to avoid warnings
        input_array = df_input.to_numpy()

        # Predict using the RandomForest model
        prediction = model.predict(input_array)

        # Map the prediction output to the correct loan status
        loan_status = ' Approved' if prediction[0] == 1 else ' Rejected'

        # Generate a new loan_id
        if 'loan_id' in dataset.columns:
            new_loan_id = dataset['loan_id'].max() + 1
        else:
            new_loan_id = 1  # Start from 1 if loan_id column doesn't exist

        # Prepare the new record for appending to the dataset
        new_record = df_input.copy()
        new_record['loan_id'] = new_loan_id  # Add the loan_id column
        new_record['loan_status'] = loan_status  # Correctly set the loan status based on the prediction

        # Convert numerical values for education and self_employed back to their categorical values
        new_record['education'] = new_record['education'].map(education_map)
        new_record['self_employed'] = new_record['self_employed'].map(self_employed_map)

        # Rearrange columns to place loan_id first
        new_record = new_record[['loan_id'] + [col for col in new_record.columns if col != 'loan_id']]

        # Append the new record to the dataset and save it
        new_record.to_csv('loan_approval_dataset.csv', mode='a', header=False, index=False)

        return redirect(url_for('result', result=loan_status))
    except Exception as e:
        return f"An error occurred: {e}"

@app.route('/result/<result>')
def result(result):
    return render_template('result.html', result=result)

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)

