# Import required libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

# Loading the model here we are using Linear Regression model from phase 2 as it has the highest accuracy
@st.cache_data
def load_ml_model():
    with open('logistic_regression_model.pkl', 'rb') as file:
        return pickle.load(file)

# Loading the scaled values  used to normalize training data for preproccessing the user inputs
@st.cache_data
def load_scaler():
    scaler_filename = 'scaler.pkl'
    with open(scaler_filename, 'rb') as file:
        return pickle.load(file)

# Instantiating the model and the scaled values
model = load_ml_model()
scaler = load_scaler()

# Method to preprocess the user inputs, the preprocessing steps are similar to Phase 1
def preprocess_input(user_input):
    # Converting the TotalCharges from string to numerical value
    user_input['TotalCharges'] = pd.to_numeric(user_input['TotalCharges'], errors='coerce')
    
    # Replacing No phone service with No in MultipleLines column/field
    user_input['MultipleLines'] = user_input['MultipleLines'].replace('No phone service', 'No')
    
    # Converting Yes and No to 1 and 0 for binary columns
    binary_columns = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'PaperlessBilling']
    for var in binary_columns:
        user_input[var] = pd.Categorical(user_input[var], categories=['No', 'Yes'], ordered=True).codes
    user_input['gender'] = pd.Categorical(user_input['gender'], categories=['Female', 'Male'], ordered=True).codes
    
    # Feature engineering by creating a binary flag for customers with internet service and creating a new feature HasInternetService
    user_input['HasInternetService'] = (user_input['InternetService'] == 'No').astype(int)
    
    # Creating new feature TotalServices by calculating sum of the services where there value is yes
    services_columns = ['PhoneService', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    user_input['TotalServices'] = user_input[services_columns].apply(lambda row: (row == 'Yes').sum(), axis=1)
    
    # Creating new feature for AvgMonthlyCharges
    user_input['AvgMonthlyCharges'] = user_input['TotalCharges'] / user_input['tenure']
    
    # Creating a new feature HadPriceIncrease i.e. a binary flag for price increase
    user_input['HadPriceIncrease'] = (user_input['MonthlyCharges'] > user_input['AvgMonthlyCharges']).astype(int)
    
    # Creating a new feature HadPriceIncrease i.e. a binary flag for family support
    user_input['HasFamilySupport'] = ((user_input['Partner'] == 1) | (user_input['Dependents'] == 1)).astype(int)
    
    # Creating feature for high value coustomers
    user_input['IsHighValue'] = (user_input['TotalCharges'] > user_input['TotalCharges'].quantile(0.8)).astype(int)
    
    # Calculating charge difference
    user_input['ChargeDifference'] = user_input['MonthlyCharges'] - user_input['AvgMonthlyCharges']
    
    # Applying One Hot encoding for multiclass columns
    multi_categorical_columns = ['InternetService', 'OnlineSecurity', 'OnlineBackup', 'PaymentMethod', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract']
    user_input = pd.get_dummies(user_input, columns=multi_categorical_columns, prefix=multi_categorical_columns)
    
    return user_input

# Method fpr applying styles to the graphs in visualization
def apply_styles(fig, xaxis_title, yaxis_title):
    fig.update_layout(
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        plot_bgcolor="white",
        paper_bgcolor="white", 
        font=dict(color="black", size=14),
        xaxis=dict(
            title=dict(font=dict(color="black")),
            showline=True,
            linecolor='black',
            tickfont=dict(color='black')
        ),
        yaxis=dict(
            title=dict(font=dict(color="black")),
            showline=True,
            linecolor='black',
            tickfont=dict(color='black')
        ),
        title=dict(
            text=fig.layout.title.text,
            font=dict(color='black', size=16) 
        ),
        showlegend=True,
        legend=dict(
            font=dict(color='black'),
            title=dict(font=dict(color='black'))
        ),
        margin=dict(l=50, r=50, t=50, b=50),
        width=800,
        height=500,
    )
    return fig

# Custom styling for sidebar
st.markdown(
    """
    <style>
    /* Change sidebar font size globally */
    [data-testid="stSidebar"] * {
        font-size: 20px !important;  /* Apply a larger font size to all elements */
    }

    /* Optionally, adjust spacing for better appearance */
    [data-testid="stSidebar"] .css-1lcbmhc {
        padding-top: 10px !important;
        padding-bottom: 10px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar creation
st.sidebar.title("Customer Churn Prediction Tool")
menu = st.sidebar.radio("Menu", ["Home", "Single Prediction", "Batch Prediction", "Visualization"])

# Home Page
if menu == "Home":
    st.title("Welcome to the Telco Customer Churn Prediction Tool")
    st.write("This tool helps Teleco businesses predict their customer churn")
    st.write("Navigate to 'Single Prediction' or 'Batch Prediction' from the sidebar for predicting customer churn")
    st.write("Navigate to 'Visualization' to get insights of the data")

# Single user input prediction
elif menu == "Single Prediction":
    st.title("Single Customer Churn Prediction")

    st.write("Enter Customer Information:")

    senior_citizen_yes_no = {"No": 0, "Yes": 1}

    # Form fields for taking user input
    user_input = {
        "gender": st.selectbox("Gender", ["Female", "Male"]),
        "SeniorCitizen": senior_citizen_yes_no[st.selectbox("Senior Citizen", ["No", "Yes"])],
        "Partner": st.selectbox("Partner", ["No", "Yes"]),
        "Dependents": st.selectbox("Dependents", ["No", "Yes"]),
        "tenure": st.number_input("Tenure (Months)", min_value=0, max_value=100, value=1),
        "PhoneService": st.selectbox("Phone Service", ["No", "Yes"]),
        "MultipleLines": st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"]),
        "InternetService": st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"]),
        "OnlineSecurity": st.selectbox("Online Security", ["No", "Yes"]),
        "OnlineBackup": st.selectbox("Online Backup", ["No", "Yes"]),
        "DeviceProtection": st.selectbox("Device Protection", ["No", "Yes"]),
        "TechSupport": st.selectbox("Tech Support", ["No", "Yes"]),
        "StreamingTV": st.selectbox("Streaming TV", ["No", "Yes"]),
        "StreamingMovies": st.selectbox("Streaming Movies", ["No", "Yes"]),
        "Contract": st.selectbox("Contract", ["Month-to-month", "One year", "Two year"]),
        "PaperlessBilling": st.selectbox("Paperless Billing", ["No", "Yes"]),
        "PaymentMethod": st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]),
        "MonthlyCharges": st.number_input("Monthly Charges", min_value=0.0, value=50.0),
        "TotalCharges": st.text_input("Total Charges", value="500")
    }

    # Converting user input tp pandas dataframe
    user_input_df = pd.DataFrame([user_input])

    if st.button("Predict"):
        try:
            # Preprocessing user input data by calling preprocess_input function
            preprocessed_input_data = preprocess_input(user_input_df)

            # Checking if all the columns required for prediction are available or not
            required_columns = scaler.feature_names_in_
            missing_cols = [col for col in required_columns if col not in preprocessed_input_data.columns]
            for col in missing_cols:
                preprocessed_input_data[col] = 0

            preprocessed_input_data = preprocessed_input_data[required_columns]

            # Normalizing this data
            scaled_data = scaler.transform(preprocessed_input_data)

            # Predicting the customer churn
            pred = model.predict(scaled_data)[0]
            probability = model.predict_proba(scaled_data)[0]

            pred_op = ''
            if pred == 1:
                pred_op = 'Churn'
            else:
                pred_op = 'No Churn'

            st.write(f"**Prediction:** {pred_op}")
            st.write(f"**Churn Probability:** {probability[1]:.2f}")
            st.write(f"**No Churn Probability:** {probability[0]:.2f}")
        except Exception as e:
            st.error(f"Error in prediction: {e}")

# Batch user input prediction
elif menu == "Batch Prediction":
    st.title("Batch Customer Churn Prediction")

    # File upload option to user
    uploaded_input_file = st.file_uploader("Upload CSV file", type="csv")
    
    if uploaded_input_file:
        user_input_batch_data = pd.read_csv(uploaded_input_file)
        st.write("Uploaded Data:")
        st.dataframe(user_input_batch_data.head())
        
        if st.button("Predict"):
            try:
                # Check if user has given empty values in the columns
                user_input_batch_data = user_input_batch_data.dropna()

                if user_input_batch_data.empty:
                    st.error("No valid rows available for prediction Re upload file with valid data")
                else:
                    # Preprocess batch data by calling preprocess_input function
                    preprocessed_batch_data = preprocess_input(user_input_batch_data)

                    preprocessed_batch_data = preprocessed_batch_data.dropna()
                    if preprocessed_batch_data.empty:
                        st.error("No valid rows available for prediction Re upload file with valid data")
                    else:
                        # Checking if all the columns required for prediction are available or not
                        required_columns = scaler.feature_names_in_
                        missing_cols = [col for col in required_columns if col not in preprocessed_batch_data.columns]
                        for col in missing_cols:
                            preprocessed_batch_data[col] = 0

                        preprocessed_batch_data = preprocessed_batch_data[required_columns]

                        # Normalizing this data
                        scaled_batch = scaler.transform(preprocessed_batch_data)

                        # Predict the customer churn for all the data
                        pred = model.predict(scaled_batch)
                        probabilities = model.predict_proba(scaled_batch)

                        user_input_batch_data = user_input_batch_data.loc[preprocessed_batch_data.index]
                        user_input_batch_data['Churn Prediction'] = pred
                        user_input_batch_data['Churn Probability'] = probabilities[:, 1]
                        user_input_batch_data['Churn Prediction'] = user_input_batch_data['Churn Prediction'].map({1: "Churn", 0: "No Churn"})

                        st.write("Predictions:")
                        st.dataframe(user_input_batch_data)

                        # Download option for users
                        csv = user_input_batch_data.to_csv(index=False)
                        st.download_button("Download", data=csv, file_name="predictions.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Error in batch prediction: {e}")

# Visualization
elif menu == "Visualization":
    st.title("Data Visualizations and Insights")

    # Load Preprocessed Data for visualization
    @st.cache_data
    def load_preprocessed_input_data():
        return pd.read_csv('encoded_customer_churn_.csv')

    df = load_preprocessed_input_data()

    st.subheader("Key Metrics")
    churn_rate = df['Churn'].mean() * 100
    st.metric("Overall Churn Rate", f"{churn_rate:.2f}%")

    # Adding filter for visualization
    churn_filter = st.radio("Filter by Churn:", options=["All", "Churn", "No Churn"])
    if churn_filter == "Churn":
        df = df[df['Churn'] == 1]
    elif churn_filter == "No Churn":
        df = df[df['Churn'] == 0]

    # Plotting histogram for Monthly Charges
    st.subheader("Distribution of Monthly Charges")
    fig = px.histogram(
        df,
        x="MonthlyCharges",
        nbins=30,
        title="Monthly Charges Distribution",
        marginal="box",
        hover_data=df.columns,
        color_discrete_sequence=["#1f77b4"],
    )
    fig = apply_styles(fig, xaxis_title="Monthly Charges", yaxis_title="Count")
    st.plotly_chart(fig)

    # Plotting scatter plot for Tenure vs. Monthly Charges
    st.subheader("Tenure vs. Monthly Charges")
    fig = px.scatter(
        df,
        x="tenure",
        y="MonthlyCharges",
        color="Churn",
        title="Tenure vs. Monthly Charges",
        hover_data=["TotalCharges", "Contract"],
        color_discrete_map={0: "blue", 1: "red"},
    )
    fig = apply_styles(fig, xaxis_title="Tenure", yaxis_title="Monthly Charges")
    st.plotly_chart(fig)

    # plottong box plot for Services and Churn Correlation
    st.subheader("Correlation of Total Services with Churn")
    fig = px.box(
        df,
        x="Churn",
        y="TotalServices",
        color="Churn",
        title="Total Services vs. Churn",
        color_discrete_map={0: "blue", 1: "red"},
        hover_data=df.columns,
    )
    fig = apply_styles(fig, xaxis_title="Churn", yaxis_title="Total Services")
    st.plotly_chart(fig)

    # Feature of exploring Specific Features Section
    st.subheader("Explore Specific Features")
    feature = st.selectbox("Choose a feature to visualize:", options=df.columns)

    if feature:
        if feature in ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'SeniorCitizen']:
            mapping = {0: "No", 1: "Yes"}
            if feature == 'gender':
                mapping = {0: "Female", 1: "Male"}
            df[feature] = df[feature].map(mapping)

        # Plot bar chart for categorical data
        if df[feature].dtype == "object" or feature in ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'SeniorCitizen']:
            feature_counts = df[feature].value_counts().reset_index()
            feature_counts.columns = [feature, 'count']
            
            fig = px.bar(
                feature_counts,
                x=feature,
                y='count',
                title=f"Distribution of {feature.capitalize()}",
                labels={feature: feature.capitalize(), 'count': "Count"},
                color=feature,
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig = apply_styles(fig, xaxis_title=feature.capitalize(), yaxis_title="Count")
            st.plotly_chart(fig)
        # Plot histogram for numerical data
        else:
            fig = px.histogram(
                df,
                x=feature,
                title=f"Distribution of {feature.capitalize()}",
                marginal="box",
                hover_data=df.columns,
                color_discrete_sequence=["#ff7f0e"],
            )
            fig = apply_styles(fig, xaxis_title=feature.capitalize(), yaxis_title="Count")
            st.plotly_chart(fig)

# -------------------------------------------------------------------------------------
# References
# -------------------------------------------------------------------------------------
# https://docs.streamlit.io/get-started/fundamentals/main-concepts
# https://docs.streamlit.io/get-started/fundamentals/advanced-concepts
# https://docs.streamlit.io/get-started/tutorials/create-an-app
# https://pandas.pydata.org/docs/reference/api/pandas.to_numeric.html
# https://pandas.pydata.org/docs/reference/api/pandas.Categorical.html
# https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html
# https://docs.python.org/3/library/pickle.html
# https://discuss.streamlit.io/t/applying-custom-css-to-manually-created-containers/33428/2
# https://plotly.com/python/bar-charts/
# https://plotly.com/python/histograms/
# https://plotly.com/python/line-and-scatter/
# https://plotly.com/python/box-plots/
# https://docs.streamlit.io/develop/api-reference/charts/st.plotly_chart
# https://docs.streamlit.io/develop/api-reference/widgets/st.selectbox
# https://docs.streamlit.io/develop/api-reference/widgets/st.text_input
# https://www.geeksforgeeks.org/how-to-change-the-position-of-legend-using-plotly-python/