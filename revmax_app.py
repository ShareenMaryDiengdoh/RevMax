import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data/sales_data.csv", parse_dates=["Date"])
    return df

df = load_data()

# Sidebar Filters
st.sidebar.title("🔍 Filter Sales Data")
start_date = st.sidebar.date_input("Start Date", df['Date'].min())
end_date = st.sidebar.date_input("End Date", df['Date'].max())
category = st.sidebar.multiselect("Select Category", df['Category'].unique(), default=df['Category'].unique())
store = st.sidebar.multiselect("Select Store", df['StoreLocation'].unique(), default=df['StoreLocation'].unique())

# Filter Data
filtered_df = df[
    (df['Date'] >= pd.to_datetime(start_date)) &
    (df['Date'] <= pd.to_datetime(end_date)) &
    (df['Category'].isin(category)) &
    (df['StoreLocation'].isin(store))
]

# Title
st.title("📊 RevMax: Revenue Maximization Dashboard")
st.markdown("Get insights on product performance, promotions, and sales trends.")

# KPIs
total_revenue = filtered_df['Revenue'].sum()
total_units = filtered_df['UnitsSold'].sum()
top_product = filtered_df.groupby('ProductName')['Revenue'].sum().idxmax()
top_store = filtered_df.groupby('StoreLocation')['Revenue'].sum().idxmax()

col1, col2, col3, col4 = st.columns(4)
col1.metric("💰 Total Revenue", f"₹{total_revenue:,.2f}")
col2.metric("📦 Total Units Sold", f"{total_units:,}")
col3.metric("🏆 Top Product", top_product)
col4.metric("🏪 Best Store", top_store)

# --- Sales Over Time ---
st.subheader("📈 Revenue Over Time")
revenue_over_time = filtered_df.groupby('Date')['Revenue'].sum().reset_index()
st.line_chart(revenue_over_time.set_index('Date'))

# --- Top Products ---
st.subheader("🥇 Top 5 Products by Revenue")
top_products = filtered_df.groupby('ProductName')['Revenue'].sum().sort_values(ascending=False).head(5)
st.bar_chart(top_products)

# --- Sales by Category ---
st.subheader("📦 Revenue by Category")
cat_rev = filtered_df.groupby('Category')['Revenue'].sum()
fig1, ax1 = plt.subplots()
ax1.pie(cat_rev, labels=cat_rev.index, autopct='%1.1f%%', startangle=90)
ax1.axis('equal')
st.pyplot(fig1)

# --- Promotion Analysis ---
st.subheader("💡 Impact of Promotions")
promo_df = filtered_df.groupby('PromotionApplied')['Revenue'].sum()
st.bar_chart(promo_df)

# --- Raw Data (optional toggle) ---
with st.expander("🔍 View Raw Data"):
    st.dataframe(filtered_df.head(50))
# --- Demand Prediction Section ---
st.subheader("🔮 Demand Prediction (Units Sold)")

# Encode categorical features
df_encoded = filtered_df.copy()
le_promo = LabelEncoder()
le_cat = LabelEncoder()
le_store = LabelEncoder()

df_encoded['PromotionApplied'] = le_promo.fit_transform(df_encoded['PromotionApplied'])
df_encoded['Category'] = le_cat.fit_transform(df_encoded['Category'])
df_encoded['StoreLocation'] = le_store.fit_transform(df_encoded['StoreLocation'])

X = df_encoded[['Price', 'PromotionApplied', 'Category', 'StoreLocation']]
y = df_encoded['UnitsSold']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
score = model.score(X_test, y_test)

st.markdown(f"Model R² Score: `{score:.2f}`")

# Predict units sold for any input
st.markdown("### 📊 Predict Units Sold")
col1, col2 = st.columns(2)
input_price = col1.number_input("Enter Price", min_value=1.0, value=50.0)
input_promo = col1.selectbox("Promotion Applied?", ['Yes', 'No'])
input_cat = col2.selectbox("Category", df['Category'].unique())
input_store = col2.selectbox("Store Location", df['StoreLocation'].unique())

encoded_input = [[
    input_price,
    le_promo.transform([input_promo])[0],
    le_cat.transform([input_cat])[0],
    le_store.transform([input_store])[0]
]]

predicted_units = model.predict(encoded_input)[0]
st.success(f"📦 Predicted Units Sold: **{predicted_units:.1f}**")
# --- Product Recommendations (Bundling) ---
st.subheader("🧠 Product Bundling Suggestions")

# Sample co-purchase simulation: build a transaction dataset
# For demo: group daily sales by store, use product names as basket items
trans_df = df.groupby(['Date', 'StoreLocation'])['ProductName'].apply(list).reset_index()

# One-hot encode product purchases
from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()
te_result = te.fit(trans_df['ProductName']).transform(trans_df['ProductName'])
basket_df = pd.DataFrame(te_result, columns=te.columns_)

# Frequent itemsets
frequent_items = apriori(basket_df, min_support=0.02, use_colnames=True)
rules = association_rules(frequent_items, metric="lift", min_threshold=1.2)

# Show Top Rules
if not rules.empty:
    st.markdown("### 🔗 Frequently Bought Together:")
    top_rules = rules.sort_values(by='lift', ascending=False)[['antecedents', 'consequents', 'lift']].head(5)
    for _, row in top_rules.iterrows():
        st.write(f"✅ **{', '.join(list(row['antecedents']))}** ➡ **{', '.join(list(row['consequents']))}** (Lift: {row['lift']:.2f})")
else:
    st.info("Not enough transactions for bundling suggestions. Try with more data.")