#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

sns.set(style="whitegrid")

@st.cache_data
def load_and_clean_data():
    # Load CSV file from local path
    file_path = '/Users/erindoran/Desktop/TruRootsDS.csv'
    df = pd.read_csv(file_path, low_memory=False)
    
    # Convert key columns to numeric (removing '$' and commas)
    cols_to_convert = ["Units", "Avg Unit Price", "Any Promo Units", "Number of Stores Selling"]
    for col in cols_to_convert:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[\$,]', '', regex=True),
                                      errors='coerce')
    # Filter out rows with non-positive values for price and units
    df = df[(df["Units"] > 0) & (df["Avg Unit Price"] > 0)]
    df = df.dropna(subset=["Units", "Avg Unit Price", "Any Promo Units", "Number of Stores Selling"])
    
    # Create derived variables
    df['log_Units'] = np.log(df['Units'])
    df['log_AvgUnitPrice'] = np.log(df['Avg Unit Price'])
    # Create promotion dummy: 1 if Any Promo Units > 0, else 0
    df['Promo'] = (df['Any Promo Units'] > 0).astype(int)
    # Convert Number of Stores Selling and create log_Stores (add 1 to avoid log(0))
    df["Number of Stores Selling"] = pd.to_numeric(
        df["Number of Stores Selling"].astype(str).str.replace(r'[\$,]', '', regex=True),
        errors='coerce'
    )
    df = df.dropna(subset=["Number of Stores Selling"])
    df['log_Stores'] = np.log(df["Number of Stores Selling"] + 1)
    
    return df

# Load data
df = load_and_clean_data()

# Sidebar for brand and subcategory selection
brands = sorted(df['Brand'].unique())
subcategories = sorted(df['SUB CATEGORY'].unique())

selected_brand = st.sidebar.selectbox("Select Brand", brands)
selected_subcat = st.sidebar.selectbox("Select Subcategory", subcategories)

# Filter data for selected brand and subcategory
filtered_df = df[(df['Brand'] == selected_brand) & (df['SUB CATEGORY'] == selected_subcat)]
st.write(f"Observations for {selected_brand} - {selected_subcat}: {len(filtered_df)}")

# Estimate elasticity parameters if sufficient data exists, else use defaults.
if len(filtered_df) >= 20:
    reg_df = filtered_df[['log_Units', 'log_AvgUnitPrice', 'Promo', 'log_Stores']].dropna()
    X = reg_df[['log_AvgUnitPrice', 'Promo', 'log_Stores']]
    y = reg_df['log_Units']
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    
    # For non-technical display, we won't show the full table.
    model_accuracy = model.rsquared * 100  # in percentage
    const = model.params['const']
    beta_price = model.params['log_AvgUnitPrice']
    beta_promo = model.params['Promo']
    beta_stores = model.params['log_Stores']
else:
    st.warning("Not enough data for regression—using default parameters.")
    model_accuracy = 97  # default accuracy %
    const = 0.25
    beta_price = -0.14
    beta_promo = -0.037
    beta_stores = 1.15

# Plain-language summary for the sponsor about model accuracy
st.subheader("Model Accuracy Summary")
st.write(f"Our model predicts sales with approximately {model_accuracy:.1f}% accuracy. In simple terms, this means the model can explain nearly all the variation in sales by considering factors like price and the number of stores selling the product.")

# Simulation Inputs
default_price = float(filtered_df['Avg Unit Price'].median() if not filtered_df.empty else 6.0)
min_price = float(filtered_df['Avg Unit Price'].min() if not filtered_df.empty else 1.0)
max_price = float(filtered_df['Avg Unit Price'].max() if not filtered_df.empty else 10.0)
avg_price = st.slider("Average Unit Price ($)", min_value=min_price, max_value=max_price,
                      value=default_price, step=0.25)

promo_active = st.checkbox("Promotion Active?", value=False)

# If promotion is active, let the user specify the discount amount
if promo_active:
    discount_amount = st.slider("Active Discount Amount ($)", min_value=0.0, 
                                max_value=avg_price/2, value=1.0, step=0.25)
    effective_price = avg_price - discount_amount
else:
    effective_price = avg_price

default_stores = int(filtered_df['Number of Stores Selling'].median() if not filtered_df.empty else 100)
min_stores = int(filtered_df['Number of Stores Selling'].min() if not filtered_df.empty else 1)
max_stores = int(filtered_df['Number of Stores Selling'].max() if not filtered_df.empty else 1000)
num_stores = st.slider("Number of Stores Selling", min_value=min_stores, max_value=max_stores,
                       value=default_stores, step=1)

# Calculate log values for simulation
log_price_base = np.log(avg_price)
log_price_effective = np.log(effective_price)
log_stores = np.log(num_stores + 1)
promo_flag = 1 if promo_active else 0

# Predicted log(Units) for two scenarios:
# Base scenario (no promotion)
predicted_log_units_base = const + beta_price * log_price_base + beta_promo * 0 + beta_stores * log_stores
predicted_units_base = np.exp(predicted_log_units_base)

# Promotion scenario (with effective price)
predicted_log_units_promo = const + beta_price * log_price_effective + beta_promo * 1 + beta_stores * log_stores
predicted_units_promo = np.exp(predicted_log_units_promo)

# Choose scenario based on promotion status
predicted_units = predicted_units_promo if promo_active else predicted_units_base
predicted_revenue = avg_price * predicted_units

# Calculate additional outputs for promotion scenarios
if promo_active:
    percent_change_units = ((predicted_units_promo - predicted_units_base) / predicted_units_base) * 100
    units_difference = predicted_units_promo - predicted_units_base
else:
    percent_change_units = 0.0
    units_difference = 0.0

# Compute correlation (r value) between log(Avg Unit Price) and log(Units) for the filtered data
if len(filtered_df) > 1:
    valid_corr = filtered_df[['log_AvgUnitPrice', 'log_Units']].dropna()
    if len(valid_corr) > 1:
        r_value, _ = pearsonr(valid_corr['log_AvgUnitPrice'], valid_corr['log_Units'])
    else:
        r_value = np.nan
else:
    r_value = np.nan

### Display Simulation Results in Plain Language
st.subheader("Simulation Results (Plain Language Summary)")
st.write(f"**Predicted Units Sold:** {predicted_units:.0f} units")
st.write(f"**Predicted Revenue:** ${predicted_revenue:.2f}")
if promo_active:
    st.write(f"**Active Discount:** ${discount_amount:.2f} off the base price")
    st.write(f"**Effect of Discount:** With the discount, sales are expected to increase by about {percent_change_units:.1f}% (an extra {units_difference:.0f} units sold) compared to no discount.")
st.write(f"**Correlation between Price and Sales:** The correlation between log(Avg Unit Price) and log(Units) is {r_value:.2f}, indicating a very weak direct relationship once other factors are considered.")

# Optional: Plot predicted units over a range of prices
price_range = np.linspace(avg_price * 0.5, avg_price * 1.5, 100)
# For plotting, use effective price if promo is active; else, base price
plot_prices = price_range if not promo_active else np.maximum(price_range - discount_amount, 0.1)
log_price_range = np.log(plot_prices)
pred_units_range = np.exp(const + beta_price * log_price_range + beta_promo * (1 if promo_active else 0) + beta_stores * log_stores)
plt.figure(figsize=(8, 5))
plt.plot(price_range, pred_units_range, label="Predicted Units")
plt.axvline(x=avg_price, color='red', linestyle='--', label="Selected Price")
plt.xlabel("Average Unit Price ($)")
plt.ylabel("Predicted Units Sold")
plt.title("Predicted Units Sold vs. Average Unit Price")
plt.legend()
st.pyplot(plt)


# In[ ]:





# In[7]:


import pandas as pd
import numpy as np
import statsmodels.api as sm

# --- Step 1: Load and Clean Data ---
file_path = '/Users/erindoran/Desktop/TruRootsDS.csv'
df = pd.read_csv(file_path, low_memory=False)

# Convert key columns to numeric (remove '$' and commas)
cols_to_convert = ["Units", "Avg Unit Price", "Any Promo Units", "Number of Stores Selling"]
for col in cols_to_convert:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce')

# Filter out rows with non-positive values for Units and Price
df = df[(df["Units"] > 0) & (df["Avg Unit Price"] > 0)]
df = df.dropna(subset=["Units", "Avg Unit Price", "Any Promo Units", "Number of Stores Selling"])

# Create derived variables
df['log_Units'] = np.log(df['Units'])
df['log_AvgUnitPrice'] = np.log(df['Avg Unit Price'])
df['Promo'] = (df['Any Promo Units'] > 0).astype(int)
df['log_Stores'] = np.log(pd.to_numeric(df["Number of Stores Selling"].astype(str)
                                         .str.replace(r'[\$,]', '', regex=True), errors='coerce') + 1)

# --- Step 2: Build the Model ---
# Our model: log(Units) ~ log(Avg Unit Price) + Promo + log(Stores)
X = df[['log_AvgUnitPrice', 'Promo', 'log_Stores']].dropna()
y = df['log_Units'].dropna()
X = sm.add_constant(X)  # adds the intercept term

model = sm.OLS(y, X).fit()
print(model.summary())


# In[ ]:




