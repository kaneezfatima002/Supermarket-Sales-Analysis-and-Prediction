import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load Dataset
# st.title("Supermarket Sales Analysis and Prediction")
st.markdown("<h1 style='text-align: center; font-size:35px;background-color: #add8e6; border-radius: 10px;'>🛒 Supermarket Sales Analysis and Prediction 📊</h1>", unsafe_allow_html=True)

st.sidebar.header("Project Options")

 # Sidebar Navigation
section = st.sidebar.radio("Navigate", [ "Introduction","EDA", "Data Preprocessing", "Model Training & Evaluation", "Conclusion"])


if section == "Introduction":
    st.image("C:/Users/M.T/Pictures/pic1.jpg")
    st.markdown("<h1 style='text-align: center; font-size:35px; '>Introduction</h1>", unsafe_allow_html=True)

    st.markdown("""
    In this project, we dive into a dataset that captures supermarket sales transactions, 
    providing insights into customer demographics, product categories, and sales performance.
    Our objectives include performing thorough exploratory data analysis (EDA) to identify key trends and 
    patterns, preprocessing the data to prepare it for machine learning models, and developing a
    regression model to predict total sales based on various factors such as customer characteristics
    and product information. The dataset features critical details such as customer gender, type, and
    payment methods, as well as product lines and sales totals, allowing us to explore both categorical 
    and temporal factors influencing sales. Join us as we explore the data and build predictive models to 
    better understand and forecast supermarket sales!
    """)

# File uploader
uploaded_file =  "E:/5th semester/IDS/supermarket_sales.csv"
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.markdown("<h3 style='text-align: left; font-size:20px;'>Dataset Overview</h3>", unsafe_allow_html=True)
    st.write(data.head())

    if section == "EDA":
        st.markdown("<h1 style='text-align: center; font-size:35px;'>Exploratory Data Analysis EDA</h1>", unsafe_allow_html=True)


        # Analysis 1: Summary Statistics
        st.markdown("<h3 style='text-align: left; font-size:20px;'>Summary Statistics</h3>", unsafe_allow_html=True)
        st.write(data.describe())

        # Analysis 2: Correlation Matrix
        st.markdown("<h3 style='text-align: left; font-size:20px;'>Corelation Matrix</h3>", unsafe_allow_html=True)
        
        try:
            numerical_data = data.select_dtypes(include=['number'])
            corr_matrix = numerical_data.corr()

            # Plot correlation matrix
            plt.figure(figsize=(10, 6))
            sns.heatmap(corr_matrix, annot=True, cmap='Blues', fmt='.2f')
            plt.title("Correlation Matrix")
            plt.xticks(rotation=45, fontsize=10)
            st.pyplot(plt)
        except Exception as e:
            st.error(f"Error generating correlation matrix: {e}")


        # Analysis 4: Data Types and Missing Values
        st.markdown("<h3 style='text-align: left; font-size:20px;'>Data Types and Missing Valuesx</h3>", unsafe_allow_html=True)
        
        st.write(data.info())
        st.write(data.isnull().sum())

        # Analysis 5: Outlier Detection
        st.markdown("<h3 style='text-align: left; font-size:20px;'>Outlier Detection</h3>", unsafe_allow_html=True)

        try:
            numerical_columns = data.select_dtypes(include=['number']).columns

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=data[numerical_columns], orient='h', palette='pastel', ax=ax)
            ax.set_title("Outlier Detection Across Numerical Columns")
            ax.set_xlabel("Value")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error generating outlier detection analysis: {e}")


        #Analysis 6: Feature Distribution
        st.markdown("<h3 style='text-align: left; font-size:25px;'>Feature Distribution Analysis</h3>", unsafe_allow_html=True)
       
        try:
                numerical_columns = data.select_dtypes(include=['number']).columns

                if len(numerical_columns) > 0:
                    for col in numerical_columns:
                        st.write(f"##### Distribution of {col}")
                        fig, ax = plt.subplots(figsize=(8, 4))
                        sns.histplot(data[col], kde=True, color='skyblue', bins=30, ax=ax)
                        ax.set_title(f"Distribution of {col}")
                        ax.set_xlabel(col)
                        ax.set_ylabel("Frequency")
                        plt.tight_layout()
                        st.pyplot(fig)
                else:
                    st.warning("No numerical columns available for distribution analysis.")
        except Exception as e:
                st.error(f"Error generating feature distribution analysis: {e}")


        # Analysis 7 Data Types Analysis
        st.markdown("<h3 style='text-align: left; font-size:20px;'>Data Types</h3>", unsafe_allow_html=True)
        st.write(data.dtypes)

        # Unique Value Counts for each column
        st.markdown("<h3 style='text-align: left; font-size:20px;'>Unique Value Count</h3>", unsafe_allow_html=True)
       
        try:
            unique_counts = data.nunique()
            st.write(unique_counts)
        except Exception as e:
            st.error(f"Error generating unique value counts: {e}")

         # Grouped Aggregation 1: Total Sales by Product Line
        st.markdown("<h2 style='text-align: left; font-size:25px;'>Grouped Aggregation by Category</h2>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: left; font-size:20px;'>Total Sales by Product Line</h3>", unsafe_allow_html=True)

        product_line_sales = data.groupby('Product line')['Total'].sum().sort_values(ascending=False)

        # Plotting the total sales by product line
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=product_line_sales.index, y=product_line_sales.values, palette='Set2', ax=ax)
        ax.set_title("Total Sales by Product Line")
        ax.set_xlabel("Product Line")
        ax.set_ylabel("Total Sales")
        plt.xticks(rotation=90, fontsize=8)
        st.pyplot(fig)

        # Grouped Aggregation 2: Total Sales by Gender
        st.markdown("<h3 style='text-align: left; font-size:20px;'>Total Sales by Gender</h3>", unsafe_allow_html=True)
        gender_sales = data.groupby('Gender')['Total'].sum().sort_values(ascending=False)

        # Plotting the total sales by gender
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=gender_sales.index, y=gender_sales.values, palette='muted', ax=ax)
        ax.set_title("Total Sales by Gender")
        ax.set_xlabel("Gender")
        ax.set_ylabel("Total Sales")
        st.pyplot(fig)
        
        # Grouped Aggregation by Time
        st.markdown("<h2 style='text-align: left; font-size:25px;'>Grouped Aggregation by Time</h2>", unsafe_allow_html=True)
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        st.write("Missing Date values:", data['Date'].isnull().sum())
        data = data.dropna(subset=['Date'])
        monthly_sales = data.resample('M', on='Date')['Total'].sum()
         
        # Grouped Aggregation 4: Monthly Total Sales
        st.markdown("<h3 style='text-align: left; font-size:20px;'>Monthly Total Sales</h3>", unsafe_allow_html=True)

        # Plotting the trend of Total Sales over months
        fig, ax = plt.subplots(figsize=(10, 6))
        monthly_sales.plot(ax=ax, color='skyblue', linestyle='-', marker='o')
        ax.set_title("Monthly Total Sales")
        ax.set_xlabel("Month")
        ax.set_ylabel("Total Sales")
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Grouped Aggregation 5: Weekly Total Sales
        st.markdown("<h3 style='text-align: left; font-size:20px;'>Weekly Total Sale</h3>", unsafe_allow_html=True)

        weekly_sales = data.resample('W', on='Date')['Total'].sum()

        # Plotting the weekly total sales
        fig, ax = plt.subplots(figsize=(10, 6))
        weekly_sales.plot(ax=ax, color='lightcoral', linestyle='-', marker='o')
        ax.set_title("Weekly Total Sales")
        ax.set_xlabel("Week")
        ax.set_ylabel("Total Sales")
        # plt.xticks(rotation=45)
        st.pyplot(fig)

        # Total Sales by Product Line and Customer Type
        st.markdown("<h3 style='text-align: left; font-size:20px;'>Total Sales by Product Line and Customer Type</h3>", unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='Product line', y='Total', hue='Customer type', data=data, ax=ax, palette='Set2')
        ax.set_title("Total Sales by Product Line and Customer Type")
        ax.set_xlabel("Product Line")
        ax.set_ylabel("Total Sales")
        plt.xticks(rotation=45, fontsize=8)
        st.pyplot(fig)


        # Count Plot: Interaction between Product Line and Customer Type
        st.markdown("<h3 style='text-align: left; font-size:20px;'>Interaction Between Product Line and Customer Type</h3>", unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(pd.crosstab(data['Product line'], data['Customer type']), annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title("Interaction Between Product Line and Customer Type")
        ax.set_xlabel("Customer Type")
        ax.set_ylabel("Product Line")
        st.pyplot(fig)



        # Grouped analysis: Interaction between Product Line, Gender, and Customer Type
        st.markdown("<h3 style='text-align: left; font-size:20px;'>Interaction Between Product Line, Gender, and Customer Type</h3>", unsafe_allow_html=True)

        grouped_sales = data.groupby(['Product line', 'Gender', 'Customer type'])['Total'].sum().unstack()

        # Plotting the interactions
        fig, ax = plt.subplots(figsize=(10, 6))
        grouped_sales.plot(kind='bar', stacked=True, ax=ax, colormap='Set2')
        ax.set_title("Interaction Between Product Line, Gender, and Customer Type")
        ax.set_xlabel("Product Line")
        ax.set_ylabel("Total Sales")
        plt.xticks(rotation=45, fontsize=8)
        st.pyplot(fig)


    elif section == "Data Preprocessing":
        st.markdown("<h1 style='text-align: center; font-size:35px;'>Data Processing</h1>", unsafe_allow_html=True)

        # Check for missing values
        st.markdown("<h3 style='text-align: left; font-size:20px;'>Missing Values Analysis</h3>", unsafe_allow_html=True)
         
        missing_values = data.isnull().sum()
        st.write(missing_values)

        if missing_values.sum() == 0:
            st.success("No missing values detected in the dataset.")
        else:
            st.warning("Missing values detected! Please handle them before proceeding.")

        #  Checking for Non-Numeric Columns
        st.markdown("<h3 style='text-align: left; font-size:20px;'>Checking for Non-Numeric Columns</h3>", unsafe_allow_html=True)
         
        non_numeric_columns = data.select_dtypes(exclude=['number']).columns
        if not non_numeric_columns.empty:
            st.write("Non-numeric columns detected:", non_numeric_columns.tolist())
        else:
            st.success("No non-numeric columns detected.")

        # Encoding Categorical Variables
        st.markdown("<h3 style='text-align: left; font-size:20px;'>Encoding Categorical Variables</h3>", unsafe_allow_html=True)
       

        try:
            categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if categorical_columns:
                st.write("Categorical columns to be encoded:", categorical_columns)
                data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
                st.success("Categorical columns have been successfully encoded.")
            else:
                st.success("No categorical columns found in the dataset. No encoding necessary.")
                data_encoded = data.copy() 

            # Display the encoded dataset
            st.markdown("<h4 style='text-align: left; font-size:15px;'>Encoded Dataset</h4>", unsafe_allow_html=True)
       
            st.write(data_encoded.head())

        except Exception as e:
            st.error(f"Error during encoding: {e}")
            st.stop()


        # Scaling/Normalizing Numerical Features
        st.markdown("<h3 style='text-align: left; font-size:20px;'>Scaling/Normalizing Numerical Features</h3>", unsafe_allow_html=True)
       
        try:
            numerical_columns = data_encoded.select_dtypes(include=['number']).columns

            scaler = StandardScaler()
            data_scaled_standard = data_encoded.copy()
            data_scaled_standard[numerical_columns] = scaler.fit_transform(data_encoded[numerical_columns])

            # Check for non-numeric columns after scaling
            non_numeric_columns_after_scaling = data_scaled_standard.select_dtypes(include=['object']).columns
            if not non_numeric_columns_after_scaling.empty:
                st.error(f"Non-numeric columns found in the final dataset: {non_numeric_columns_after_scaling.tolist()}")
                st.write(data_scaled_standard[non_numeric_columns_after_scaling].head())
                st.stop()
            else:
                st.success("No non-numeric columns found in the final dataset. Proceeding with model training.")

            st.markdown("<h4 style='text-align: left; font-size:15px;'>Standardized Data</h4>", unsafe_allow_html=True)
            
            st.write(data_scaled_standard.head())
        except Exception as e:
            st.error(f"Error during scaling: {e}")
            st.stop()


        # Split the dataset into training and testing sets
        try:
            X = data_scaled_standard.drop(columns=['Total'], errors='ignore')
            y = data_scaled_standard['Total']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            st.write("Training Data Shape:", X_train.shape)
            st.write("Testing Data Shape:", X_test.shape)

            # Visualization: Distribution of 'Total' in training and testing sets
            st.markdown("<h2 style='text-align: left; font-size:25px;'>Distribution of Target Variable 'Total' in Training and Testing Sets</h2>", unsafe_allow_html=True)
       
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.histplot(y_train, kde=True, color='skyblue', label='Training Set', ax=ax)
            sns.histplot(y_test, kde=True, color='lightcoral', label='Testing Set', ax=ax)
            ax.set_title("Distribution of 'Total' in Training and Testing Sets")
            ax.set_xlabel("Total Sales")
            ax.set_ylabel("Frequency")
            ax.legend()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error during train-test split or visualization: {e}")
            st.stop()



    elif section == "Model Training & Evaluation":
        st.markdown("<h1 style='text-align: center; font-size:35px;'>Supermarket Sales Analysis and Prediction</h1>", unsafe_allow_html=True)

        st.markdown("<h3 style='text-align: left; font-size:20px;'>Model Name:</h3>", unsafe_allow_html=True)
        st.write(f"Random Forest Regressor")

        #Same codes from data processing for the machine learning model processing
        # Check for missing values
        missing_values = data.isnull().sum()

        # Checking for Non-Numeric Columns
        non_numeric_columns = data.select_dtypes(exclude=['number']).columns

        try:
            categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if categorical_columns:
                data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
            else:
                data_encoded = data.copy()

        except Exception as e:
            st.stop()

        try:
            numerical_columns = data_encoded.select_dtypes(include=['number']).columns

            scaler = StandardScaler()
            data_scaled_standard = data_encoded.copy()
            data_scaled_standard[numerical_columns] = scaler.fit_transform(data_encoded[numerical_columns])

            non_numeric_columns_after_scaling = data_scaled_standard.select_dtypes(include=['object']).columns
            if not non_numeric_columns_after_scaling.empty:
                st.stop()

        except Exception as e:
            st.stop()


        # Split the dataset into training and testing sets
        try:
            X = data_scaled_standard.drop(columns=['Total'], errors='ignore')
            y = data_scaled_standard['Total']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        except Exception as e:
            st.stop()

        # Model Training and Evaluation
        try:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            # Evaluate the model
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)

            # Display evaluation metrics
            st.markdown("<h3 style='text-align: left; font-size:20px;'>Evaluation Metrics:</h3>", unsafe_allow_html=True)
            st.write(f"Training MAE: {train_mae:.2f}")
            st.write(f"Testing MAE: {test_mae:.2f}")
            st.write(f"Training RMSE: {train_rmse:.2f}")
            st.write(f"Testing RMSE: {test_rmse:.2f}")
            st.write(f"Training R²: {train_r2:.2f}")
            st.write(f"Testing R²: {test_r2:.2f}")

            #Tables for prediction vs actual
            train_results = pd.DataFrame({
            'Actual Sales (Train)': y_train.values,
            'Predicted Sales (Train)': y_pred_train,
            'Difference (Train)': y_train.values - y_pred_train
            })

            test_results = pd.DataFrame({
                'Actual Sales (Test)': y_test.values,
                'Predicted Sales (Test)': y_pred_test,
                'Difference (Test)': y_test.values - y_pred_test
            })

            # Display tables
            st.markdown("<h2 style='text-align: center; font-size:30px;'>Random Forest Regressor Results</h2>", unsafe_allow_html=True)

            st.markdown("<h3 style='text-align: left; font-size:20px;'>Training Data: Predictions vs Actual Values</h3>", unsafe_allow_html=True)
            st.write('First 5 rows of dataset')
            st.table(train_results.head(5))

            st.markdown("<h3 style='text-align: left; font-size:20px;'>Testing Data: Predictions vs Actual Values</h3>", unsafe_allow_html=True)
            st.write('First 5 rows of dataset')
            st.table(test_results.head(5))


            # Visualize predictions vs actuals
            st.markdown("<h3 style='text-align: left; font-size:20px;'>Visualization of Prediction vs Actuals</h3>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(x=y_test, y=y_pred_test, ax=ax, color='blue', alpha=0.6, label='Test Data')
            sns.lineplot(x=y_test, y=y_test, ax=ax, color='red', label='Ideal Prediction')
            ax.set_title("Prediction vs Actuals")
            ax.set_xlabel("Actual Total Sales")
            ax.set_ylabel("Predicted Total Sales")
            ax.legend()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error during model training or evaluation: {e}")


    elif section == "Conclusion":
        st.markdown("<h1 style='text-align: center; font-size:35px;'>Conclusion</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: left; font-size:20px;'>Project Takeaways</h3>", unsafe_allow_html=True)
            
        st.markdown("""
        - **Exploratory Insights**:
          - The correlation matrix revealed relationships between numerical features such as `Unit price`, `Quantity`, and `Total`.
          - Sales trends varied significantly by gender, product line, and customer type.
        - **Model Performance**:
          - The Random Forest Regressor achieved high accuracy with an R² score of ~X.
        - **Business Insights**:
          - The highest sales came from the `Fashion Accessories` product line.
          - Payment methods and customer type greatly influenced sales totals.

        This project highlights the importance of understanding customer demographics and product performance for business growth. Feel free to experiment further by enhancing the model or diving deeper into specific features!
        """)

