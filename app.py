import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(layout="wide")
st.title("Advanced Demand Forecasting Models for Retail Supply Chain Management Using Data Science and Machine Learning for Inventory Optimization")

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("xgboost_demand_forecast_model.pkl")

model = load_model()

# Sidebar upload
st.sidebar.header("📂 Upload Test CSV")
uploaded_file = st.sidebar.file_uploader("Upload your test_data_sample.csv", type=["csv"])

# Process after file upload
if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)

    st.subheader("📊 Uploaded Data Preview")
    st.dataframe(test_df.head())

    required_cols = ['store', 'item', 'year', 'month', 'day', 'dayofweek']
    if not all(col in test_df.columns for col in required_cols):
        st.error("❌ Required columns missing: " + ", ".join(required_cols))
    else:
        # Predict sales
        X_test = test_df[required_cols]
        predictions = model.predict(X_test)
        test_df['Predicted_Sales'] = predictions

        # =====================
        # 📊 Overall Monthly Sales (Actual or Predicted)
        # =====================
        st.subheader("📅 Total Sales by Month (All Data)")

        if 'actual_sales' in test_df.columns and test_df['actual_sales'].sum() > 0:
            sales_by_month = test_df.groupby('month')['actual_sales'].sum().reset_index()
            sales_col = 'actual_sales'
        else:
            sales_by_month = test_df.groupby('month')['Predicted_Sales'].sum().reset_index()
            sales_col = 'Predicted_Sales'

        fig1, ax1 = plt.subplots(figsize=(10, 5))
        sns.barplot(x='month', y=sales_col, data=sales_by_month, palette='viridis', ax=ax1)
        ax1.set_title("Monthly Sales (All Stores & Items)")
        ax1.set_xlabel("Month")
        ax1.set_ylabel("Total Sales")
        ax1.set_xticks(range(0, 12))
        ax1.set_xticklabels([
            'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
        ])
        ax1.grid(axis='y')
        st.pyplot(fig1)

        max_month = sales_by_month.loc[sales_by_month[sales_col].idxmax()]
        month_names = [
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ]
        st.success(f"🌟 Highest Sales Month (Overall): **{month_names[int(max_month['month']) - 1]}** with total sales of **{int(max_month[sales_col])}**")

        # =====================
        # 📌 Monthly Sales by Store & Item Selection
        # =====================
        st.subheader("📌 Monthly Sales by Store & Item")

        unique_stores = sorted(test_df['store'].unique())
        unique_items = sorted(test_df['item'].unique())

        selected_store = st.selectbox("Select Store", unique_stores)
        selected_item = st.selectbox("Select Item", unique_items)

        filtered_df = test_df[(test_df['store'] == selected_store) & (test_df['item'] == selected_item)]

        if 'actual_sales' in filtered_df.columns and filtered_df['actual_sales'].sum() > 0:
            monthly_sales = filtered_df.groupby('month')['actual_sales'].sum().reset_index()
            sales_col = 'actual_sales'
        else:
            monthly_sales = filtered_df.groupby('month')['Predicted_Sales'].sum().reset_index()
            sales_col = 'Predicted_Sales'

        fig2, ax2 = plt.subplots(figsize=(10, 5))
        sns.barplot(x='month', y=sales_col, data=monthly_sales, palette='magma', ax=ax2)
        ax2.set_title(f"Monthly Sales for Store {selected_store}, Item {selected_item}")
        ax2.set_xlabel("Month")
        ax2.set_ylabel("Total Sales")
        ax2.set_xticks(range(0, 12))
        ax2.set_xticklabels([
            'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
        ])
        ax2.grid(axis='y')
        st.pyplot(fig2)

        if not monthly_sales.empty:
            max_month = monthly_sales.loc[monthly_sales[sales_col].idxmax()]
            st.success(f"📈 Highest Sales Month for Store {selected_store}, Item {selected_item}: **{month_names[int(max_month['month']) - 1]}** with **{int(max_month[sales_col])}** sales.")
        else:
            st.warning("No data available for the selected store and item.")

        # =====================
        # 📈 Actual vs Predicted Chart
        # =====================
        st.subheader("📉 Actual vs Predicted Sales (First 100)")

        if 'actual_sales' in test_df.columns:
            if test_df['actual_sales'].sum() == 0:
                st.warning("⚠️ 'actual_sales' contains only zeros — showing predictions only.")
                fig3, ax3 = plt.subplots(figsize=(12, 6))
                ax3.plot(test_df['Predicted_Sales'][:100], label="Predicted", color='blue')
                ax3.set_title("Predicted Sales (First 100)")
                ax3.set_xlabel("Samples")
                ax3.set_ylabel("Sales")
                ax3.legend()
                st.pyplot(fig3)
            else:
                fig3, ax3 = plt.subplots(figsize=(12, 6))
                ax3.plot(test_df['actual_sales'][:100], label="Actual", color='green')
                ax3.plot(test_df['Predicted_Sales'][:100], label="Predicted", color='orange')
                ax3.set_title("Actual vs Predicted Sales (First 100)")
                ax3.set_xlabel("Samples")
                ax3.set_ylabel("Sales")
                ax3.legend()
                st.pyplot(fig3)

                rmse = np.sqrt(np.mean((test_df['actual_sales'] - test_df['Predicted_Sales']) ** 2))
                st.success(f"✅ RMSE on uploaded data: {rmse:.2f}")
        else:
            st.info("ℹ️ No 'actual_sales' column found.")
            fig3, ax3 = plt.subplots(figsize=(12, 6))
            ax3.plot(test_df['Predicted_Sales'][:100], label="Predicted", color='blue')
            ax3.set_title("Predicted Sales (First 100)")
            ax3.set_xlabel("Samples")
            ax3.set_ylabel("Sales")
            ax3.legend()
            st.pyplot(fig3)

        # =====================
        # 📥 Download Predicted Data
        # =====================
        st.subheader("🧾 Predicted Output Table")
        st.dataframe(test_df.head(10))

        csv = test_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Predictions as CSV",
            data=csv,
            file_name='predicted_sales.csv',
            mime='text/csv',
        )
else:
    st.info("👈 Upload a CSV file to begin.")
