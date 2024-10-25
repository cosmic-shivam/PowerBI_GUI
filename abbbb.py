import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import pickle  # For saving the model

# Load the dataset
df = pd.read_csv('loan_approval_dataset.csv')

# Remove leading and trailing spaces from column names
df.columns = df.columns.str.strip()

# Drop duplicate columns (due to some columns being listed twice)
df = df.loc[:, ~df.columns.duplicated()]

# Check the column names and types after cleaning
print("Column Names and Data Types after cleaning:")
print(df.dtypes)

# Strip spaces from object-type columns (if necessary)
if df['loan_status'].dtype == 'object':
    df['loan_status'] = df['loan_status'].str.strip()

# Drop the 'loan_id' column as it's not meaningful
df = df.drop(columns=['loan_id'])

# Handle missing values in loan_status
print("Checking for missing values in 'loan_status':")
print(df['loan_status'].isnull().sum())

# Drop rows with NaN values in loan_status
df = df.dropna(subset=['loan_status'])

# Handle missing values in 'education' and 'self_employed'
# Fill missing values with the most frequent category (mode)
df['education'].fillna(df['education'].mode()[0], inplace=True)
df['self_employed'].fillna(df['self_employed'].mode()[0], inplace=True)

# Encoding categorical variables
df['education'] = df['education'].map({'Graduate': 1, 'Not Graduate': 0})
df['self_employed'] = df['self_employed'].map({'Yes': 1, 'No': 0})
df['loan_status'] = df['loan_status'].map({'Approved': 1, 'Rejected': 0})

# After dropping NaN values, check if there are any remaining NaNs
print("Checking for missing values in the DataFrame after encoding:")
print(df.isnull().sum())

# Fill missing values only for numeric columns with their respective mean values
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Split the data into features (X) and target (y)
X = df.drop(['loan_status'], axis=1)  # Features
y = df['loan_status']  # Target

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, 
                           cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best estimator from grid search
best_rf_model = grid_search.best_estimator_

# Save the trained model using pickle
with open('loan_approval_model.pkl', 'wb') as model_file:
    pickle.dump(best_rf_model, model_file)

# Predictions for model evaluation
y_pred = best_rf_model.predict(X_test)
y_pred_prob = best_rf_model.predict_proba(X_test)[:, 1]

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print evaluation results
print("Best Parameters from Grid Search:", grid_search.best_params_)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)
print("Confusion Matrix:\n", conf_matrix)

# Plotting feature importances
importances = best_rf_model.feature_importances_
feature_names = df.drop(['loan_status'], axis=1).columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance')
plt.show()

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr, color='orange', label='ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
