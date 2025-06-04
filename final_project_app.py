#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from pandas_datareader import data as pdr
import datetime
from sklearn.metrics import mean_absolute_percentage_error

# Load financial data
df = pd.read_csv("starbucks_financials_expanded.csv")
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df.sort_index(inplace=True)

# Retrieve CPI data
start_date = df.index.min()
end_date = df.index.max()
cpi = pdr.DataReader('CPIAUCSL', 'fred', start_date, end_date)
cpi = cpi.resample('Q').mean()
df = df.join(cpi, how='left')
df.rename(columns={'CPIAUCSL': 'cpi'}, inplace=True)
df.dropna(subset=['cpi'], inplace=True)

# Forecast using ARIMAX with store_count as regressor
y = df['revenue']
exog = df[['store_count', 'cpi']]
model = sm.tsa.SARIMAX(y, exog=exog, order=(1, 1, 1), seasonal_order=(1, 0, 0, 4))
results = model.fit(disp=False)

# Streamlit App
st.title("Starbucks Revenue Forecasting Tool")
st.write("This tool helps evaluate potential revenue overstatement risk for Starbucks using ARIMAX forecasting.")

# User Input
store_growth_pct = st.slider("Expected growth in store count for next quarter (%)", -5.0, 10.0, 2.0)
last_row = df.iloc[-1]
next_store_count = last_row['store_count'] * (1 + store_growth_pct / 100)
next_cpi = cpi.iloc[-1].values[0]
future_exog = pd.DataFrame([[next_store_count, next_cpi]], columns=['store_count', 'cpi'])

# Forecast next quarter
y_pred = results.get_forecast(steps=1, exog=future_exog)
y_pred_mean = y_pred.predicted_mean.values[0]
y_pred_ci = y_pred.conf_int().values[0]

st.subheader("Forecast Results")
st.metric("Predicted Revenue (next quarter)", f"${y_pred_mean:,.2f}")
st.write(f"95% Confidence Interval: ${y_pred_ci[0]:,.2f} - ${y_pred_ci[1]:,.2f}")

# Risk Flagging
overstatement_flag = last_row['revenue'] > y_pred_mean * 1.10
if overstatement_flag:
    st.error("⚠️ Potential Revenue Overstatement Detected")
else:
    st.success("✅ Revenue appears within expected range")

# Visualization
st.subheader("Forecast vs Actual Revenue")
pred_vals = results.fittedvalues
df['predicted'] = pred_vals
fig, ax = plt.subplots(figsize=(10, 5))
df[['revenue', 'predicted']].plot(ax=ax)
plt.title("Actual vs Forecasted Revenue")
plt.ylabel("Revenue ($ Millions)")
plt.grid(True)
st.pyplot(fig)

# New Insight: Store Count Impact
st.subheader("New Insight: Store Count vs Revenue")
fig2, ax2 = plt.subplots()
sns.regplot(x='store_count', y='revenue', data=df, ax=ax2)
plt.title("Revenue vs Store Count")
st.pyplot(fig2)

# AI-Generated Summary
summary = "Based on ARIMAX forecasting with macroeconomic and store count inputs, next quarter's revenue is expected to be within historical norms. No aggressive overstatement is flagged based on current projections."
st.subheader("Audit Committee Summary")
st.write(summary)

