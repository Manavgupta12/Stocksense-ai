import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import streamlit as st

# Cache data loading with ticker as part of the cache key
@st.cache_data
def load_data(ticker, start_date, end_date):
    return yf.download(ticker, start=start_date, end=end_date)

st.title('🤖 StockSense – AI-Powered Stock Price Predictor')
st.markdown("""
This tool uses **Random Forest Regression** to analyze stock trends and forecast prices for up to 30 days ahead. It supports multiple companies and provides downloadable prediction data.
""")

# User inputs with dropdown for multiple companies
companies = {
    'Vedanta Limited': 'VEDL.NS',
    'Reliance Industries': 'RELIANCE.NS', 
    'Tata Steel': 'TATASTEEL.NS'
}
days = st.slider("📆 Predict how many days ahead?", min_value=1, max_value=30, value=7)

selected_company = st.selectbox(
    'Select Company',
    options=["Select Company"] + list(companies.keys())
)

if selected_company == "Select Company":
    st.warning("Please select a company to proceed.")
    st.stop()


user_input = companies[selected_company]
start_date = '2014-01-01'
end_date = '2024-12-31'

# Load data
df = load_data(user_input, start_date, end_date)

# Basic Visualizations
st.subheader(f'{selected_company} Data Overview (2014-2024)')
st.write(df.describe())

st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.title(f'{selected_company} Closing Prices')
st.pyplot(fig)

# Feature Engineering
df['MA_50'] = df['Close'].rolling(50).mean()
df['MA_200'] = df['Close'].rolling(200).mean()
df['Daily_Return'] = df['Close'].pct_change()
df['Volatility'] = df['Close'].rolling(30).std()

# Drop NA values
# Create prediction target using days selected by user
df['Target'] = df['Close'].shift(-days)
df.dropna(inplace=True)

# Prepare data for ML
X = df[['MA_50', 'MA_200', 'Daily_Return', 'Volatility']]
y = df['Target'].values

# Split data
split = int(0.8 * len(df))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Train Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Calculate error
mse = mean_squared_error(y_test, predictions)
st.write(f"Model Mean Squared Error: {mse:.2f}")

# Visualization of predictions
st.subheader('Random Forest Predictions vs Actual Prices')
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(df.index[split:], y_test, 'b-', label='Actual Price')
ax.plot(df.index[split:], predictions, 'r--', label='Predicted Price')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.set_title(f'{selected_company} Price Prediction')
ax.legend()
st.pyplot(fig)

# Prediction Export as CSV
result_df = pd.DataFrame({
    "Date": df.index[split:],
    "Actual Price": y_test,
    "Predicted Price": predictions
})

csv = result_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Prediction CSV",
    data=csv,
    file_name=f'{selected_company}_prediction.csv',
    mime='text/csv'
)

# Feature Importance
st.subheader('Feature Importance')
importances = model.feature_importances_
features = list(X.columns)  # Explicit conversion to list of strings

fig, ax = plt.subplots(figsize=(10,5))
y_pos = np.arange(len(features))
ax.barh(y_pos, importances, align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(features)
ax.invert_yaxis()  # Top feature at top
ax.set_xlabel('Importance Score')
ax.set_title(f'{selected_company} Feature Importance')
st.pyplot(fig)