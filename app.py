import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv('creditcard.csv')
    return data

data = load_data()

# Preprocess data
X = data.drop(columns=['Class'])
y = data['Class']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle imbalanced data
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Train models
logreg = LogisticRegression(random_state=42)
logreg.fit(X_train, y_train)

mlp = MLPClassifier(random_state=42)
mlp.fit(X_train, y_train)

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# App title
st.title('Credit Card Fraud Detection')

# Sidebar
st.sidebar.header('User Input Parameters')

def user_input_features():
    v_features = [f'V{i}' for i in range(1, 29)]
    input_data = {}
    for feature in v_features:
        input_data[feature] = st.sidebar.number_input(feature, value=0.0)
    input_data['Amount'] = st.sidebar.number_input('Amount', value=0.0)
    input_data['Time'] = st.sidebar.number_input('Time', value=0.0)
    return pd.DataFrame(input_data, index=[0])

input_df = user_input_features()

# Preprocess user input
input_scaled = scaler.transform(input_df.values)

# Model selection
model = st.sidebar.selectbox('Select model', ('Logistic Regression', 'Neural Network', 'Random Forest'))

# Predict
try:
    if model == 'Logistic Regression':
        prediction = logreg.predict(input_scaled)
        prediction_proba = logreg.predict_proba(input_scaled)[:,1]
    elif model == 'Neural Network':
        prediction = mlp.predict(input_scaled)
        prediction_proba = mlp.predict_proba(input_scaled)[:,1]
    else:
        prediction = rf.predict(input_scaled)
        prediction_proba = rf.predict_proba(input_scaled)[:,1]
except Exception as e:
    st.error('Error occurred during prediction. Please check your input values.')

# Display results
if 'prediction' in locals():
    st.subheader('Prediction')
    st.write('Fraudulent' if prediction == 1 else 'Non-Fraudulent')

    st.subheader('Prediction Probability')
    st.write(prediction_proba)

# Instructions for use
st.subheader('Instructions')
st.write("""
1. Use the sidebar to input values for the features (V1 to V28, Amount, and Time).
2. Select the model you want to use for prediction.
3. The prediction result and probability will be displayed below.
""")

