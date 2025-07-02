import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st

page = st.sidebar.radio("**Navigate**", ["Prediction Tool", "About This App"])
# Sidebar Info
st.sidebar.markdown("###  Model Info")
st.sidebar.markdown("**Model:** Random Forest Regressor")
st.sidebar.markdown("**Features Used:**")
st.sidebar.markdown("- MA_50\n- MA_200\n- Daily Return\n- Volatility\n- Price Change\n- Weekday")
st.sidebar.markdown("**Algorithm Type:** Supervised Regression (Ensemble)")


if page == "ℹ️ About This App":
    st.title("ℹ️ About Stocksense AI")
    st.markdown("""
    This project uses machine learning (Random Forest) to forecast stock prices for popular Indian companies.

    **Key Features:**
    - Real-time data via Yahoo Finance
    - User-friendly visualizations
    - AI-generated insights and risk checks
    - Downloadable predictions

    📢 **Team Members:**  
    - Manav Gupta  
    - Kishan  
    - Karandeep Singh  

    Built with ❤️ using Streamlit and Python.
    """)

    st.stop()


# Cache data loading with ticker as part of the cache key
@st.cache_data
def load_data(ticker, start_date, end_date):
    return yf.download(ticker, start=start_date, end=end_date)

st.title('Stocksense – AI-Based Stock Price Predictor')
st.markdown("""
This tool uses **Random Forest Regression** to analyze stock trends and forecast prices for up to 30 days ahead. It supports multiple companies and provides downloadable prediction data.
""")
st.markdown("""
### How to Use This Tool:
1. Select a company from the dropdown list.
2. Choose how many days ahead you want a prediction.
3. View charts, AI suggestions, and risk insights.
4. Download the results or write your own notes.

*No financial knowledge needed — the app explains everything!*
""")

st.markdown("---")

# User inputs with dropdown for multiple companies
companies = {
    'Vedanta Limited': 'VEDL.NS',
    'Reliance Industries': 'RELIANCE.NS', 
    'Tata Steel': 'TATASTEEL.NS',
    'Infosys': 'INFY.NS',
    'TCS': 'TCS.NS',
    'HDFC Bank': 'HDFCBANK.NS',
    'State Bank of India': 'SBIN.NS',
    'Wipro': 'WIPRO.NS',
    'ITC': 'ITC.NS',
    'ICICI Bank': 'ICICIBANK.NS',
    'ONGC': 'ONGC.NS',
    'Bharti Airtel': 'BHARTIARTL.NS',
    'Adani Enterprises': 'ADANIENT.NS'
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

try:
    ticker_info = yf.Ticker(user_input).info
    logo_url = ticker_info.get("logo_url", "")
    if logo_url:
        st.image(logo_url, width=80)
except:
    pass  # Don't crash the app if logo isn't found


# ✅ Error handling after data fetch
try:
    df = load_data(user_input, start_date, end_date)
except Exception as e:
    st.error(f"⚠️ Failed to fetch stock data: {e}")
    st.stop()

# ✅ Final fix with safe step-by-step checks
if df is None:
    st.error("❌ No data received. Try again later.")
    st.stop()

if df.empty:
    st.error("❌ Data is empty. Try selecting another company.")
    st.stop()

if 'Close' not in df.columns:
    st.error("❌ 'Close' column missing in data. Please choose another stock.")
    st.stop()

if df['Close'].dropna().empty:
    st.error("❌ All Close values are NaN. Try again with a different date range or stock.")
    st.stop()


st.markdown("---")


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
df['Price_Change'] = df['Close'] - df['Open']
df['Weekday'] = df.index.dayofweek  # 0=Monday

df['RSI'] = df['Close'].rolling(14).apply(lambda x: (100 - (100 / (1 + (x.pct_change().mean() / x.pct_change().std())))))

if 'RSI' not in df.columns or df['RSI'].dropna().empty:
    st.error("⚠️ RSI values could not be calculated. Try selecting a different company or time window.")
    st.stop()

if df.empty or df['RSI'].isnull().all():
    st.error("⚠️ Unable to fetch valid data for this stock right now. Please try again later or choose another company.")
    st.stop()
st.subheader(" RSI Indicator")

if df['RSI'].iloc[-1] > 70:
    st.warning("📈 RSI indicates overbought levels — consider caution.")
elif df['RSI'].iloc[-1] < 30:
    st.success("📉 RSI indicates oversold — possible buying opportunity.")
else:
    st.info("RSI is neutral — no strong signal.")

# Drop NA values
# Create prediction target using days selected by user
df['Target'] = df['Close'].shift(-days)
df.dropna(inplace=True)

# Prepare data for ML
X = df[['MA_50', 'MA_200', 'Daily_Return', 'Volatility', 'Price_Change', 'Weekday']]
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

st.write(" **Actual vs Predicted sample:**")
st.dataframe(pd.DataFrame({"Actual": y_test, "Predicted": predictions}).tail(5))

# Calculate error
mse = mean_squared_error(y_test, predictions)
st.write(f"Model Mean Squared Error: {mse:.2f}")
try:
    r2 = r2_score(y_test, predictions)
    confidence = max(0, min(100, r2 * 100))  # Clamp between 0-100
    st.write(f"R² Score: {r2:.4f}")
    st.metric("**Model Confidence Score**", f"{confidence:.2f}%")
except Exception as e:
    st.error(f"R² calculation failed: {e}")
    st.metric("📈 Model Confidence Score", "N/A")
    
st.markdown("---")


# Visualization of predictions
st.subheader('📈 Predicted Stock Price vs Real Price')
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(df.index[split:], y_test, 'b-', label='Actual Price')
ax.plot(df.index[split:], predictions, 'r--', label='Predicted Price')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.set_title(f'{selected_company} Price Prediction')
ax.legend()
from datetime import datetime
st.caption(f"Prediction generated on: {datetime.now().strftime('%d %B %Y, %I:%M %p')}")
st.caption("Made with ❤️ using Streamlit and Random Forest by Team Stocksense AI")
st.pyplot(fig)

st.markdown("---")

# Residual Error Plot
st.subheader("📉 Prediction Error (Residuals)")
errors = y_test - predictions
fig2, ax2 = plt.subplots()
ax2.plot(df.index[split:], errors, color='orange', label="Prediction Error")
ax2.axhline(0, color='gray', linestyle='--')
ax2.set_title("Residuals Over Time")
ax2.set_xlabel("Date")
ax2.set_ylabel("Error")
ax2.legend()
st.pyplot(fig2)

st.markdown("---")

# Analyze change between last actual and predicted value
change = predictions[-1] - y_test[-1]
percent_change = (change / y_test[-1]) * 100

st.subheader("Forecast Summary")

if percent_change < -2:
    if df['RSI'].iloc[-1] > 70:
        suggestions = "⚠️ Drop predicted. RSI also suggests overbought — correction likely."
    elif df['Volatility'].iloc[-1] > 0.03:
        suggestions = "⚠️ Drop predicted amidst high volatility — brace for impact."
    else:
        suggestions = "⚠️ The model predicts a downward trend, possibly due to weakening momentum."
    st.warning(suggestions)

elif percent_change > 2:
    if df['RSI'].iloc[-1] < 30:
        suggestions = "✅ Price may rise — RSI suggests it's oversold, strong bounce possible."
    elif df['Daily_Return'].mean() > 0:
        suggestions = "✅ Model shows bullish signs supported by recent positive returns."
    else:
        suggestions = "✅ Slight rise predicted — model hints mild upside in short term."
    st.success(suggestions)

else:
    if df['Volatility'].iloc[-1] < 0.01:
        suggestions = "↔️ No big moves expected — market seems sleepy with low volatility."
    else:
        suggestions = "↔️ Consolidation zone — mixed signals, wait for clear direction."
    st.info(suggestions)

st.markdown(f"📌 **Forecasted {days}-day trend** for **{selected_company}**: {suggestions}")

# Prediction Export as CSV
result_df = pd.DataFrame({
    "Date": df.index[split:],
    "Actual Price": y_test,
    "Predicted Price": predictions,
})

csv = result_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Prediction CSV",
    data=csv,
    file_name=f'{selected_company}_prediction.csv',
    mime='text/csv'
)

st.markdown("---")

volatility = df['Volatility'].iloc[-1]
avg_return = df['Daily_Return'].mean()

st.subheader("Key Metrics")
col1, col2 = st.columns(2)
col1.metric("Volatility", f"{volatility:.4f}")
col2.metric("Avg Daily Return", f"{avg_return:.4f}")

st.markdown("---")

st.subheader("Risk Check – AI Caution")

if df['Volatility'].iloc[-1] > 0.02:
    st.warning("⚠️ High market volatility detected. Be cautious with short-term trades.")
if df['Daily_Return'].mean() < 0:
    st.info("ℹ️ Average returns are negative over the selected period. Long-term hold may be safer.")

st.markdown("---")

st.subheader("Your Notes")
user_notes = st.text_area("Add your personal thoughts, trade ideas, or interpretations.")

# Save user notes to a file
st.download_button("💾 Download Your Notes", user_notes.encode('utf-8'), file_name=f"{selected_company}_notes.txt")

st.markdown("---")

# Feature Importance
st.subheader('What the AI Thinks Matters Most')
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
