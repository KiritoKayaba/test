import streamlit as st
import pickle
import pandas as pd

# Load the trained model
with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Function to preprocess the data
def preprocess_data(data):
    data['Education'] = data['Education'].map({'Post Graduate': 1, 'Under Graduate': 0})
    data['Marital_Status'] = data['Marital_Status'].map({'Relationship': 1, 'Single': 0})
    return data

# Streamlit app
def main():
    st.title('Customer Segmentation Prediction')

    # User input
    st.header('User Input')
    education = st.selectbox('Education', ['Post Graduate', 'Under Graduate'])
    marital_status = st.selectbox('Marital Status', ['Relationship', 'Single'])
    income = st.number_input('Income', min_value=0, max_value=100000)
    children = st.number_input('Children', min_value=0, max_value=10)
    expenditure = st.number_input('Expenditure', min_value=0, max_value=2000)
    overall_accepted_cmp = st.number_input('Overall Accepted Campaigns', min_value=0, max_value=10)
    num_total_purchases = st.number_input('Number of Total Purchases', min_value=0, max_value=30)
    customer_age = st.number_input('Customer Age', min_value=0, max_value=100)
    customer_shop_days = st.number_input('Customer Shop Days', min_value=0, max_value=365)

    # Make prediction button
    if st.button('Predict'):
        # Create a dataframe with user input
        input_data = pd.DataFrame({
            'Education': [education],
            'Marital_Status': [marital_status],
            'Income': [income],
            'Children': [children],
            'Expenditure': [expenditure],
            'Overall_Accepted_Cmp': [overall_accepted_cmp],
            'NumTotalPurchases': [num_total_purchases],
            'Customer_Age': [customer_age],
            'Customer_Shop_Days': [customer_shop_days]
        })

        # Preprocess the input data
        input_data = preprocess_data(input_data)

        # Make prediction
        prediction = model.predict(input_data)

        # Display result
        st.write("### Prediction:")
        st.write(prediction)

if __name__ == '__main__':
    main()
