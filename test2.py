import streamlit as st  
import pandas as pd
import altair as alt
import os
import plotly.express as px
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report

# 加載數據的函數
@st.cache_data
def load_data():
    return pd.read_csv('shopping_behavior.csv')

# 加載數據
df = load_data()

# 行為模式與預測（簡單的機器學習示例）
st.title("Predict purchasing behavior")

# 平均購買金額
avg_purchase_amount = df['Purchase Amount (USD)'].mean()
    
# 邏輯回歸模型來預測購買金額
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

# 商品與支付方式的數字化
item_encoder = LabelEncoder()
df['Item Purchased'] = item_encoder.fit_transform(df['Item Purchased'])
payment_encoder = LabelEncoder()
df['Payment Method'] = payment_encoder.fit_transform(df['Payment Method'])

# 假設資料集中有一個 Brand 欄位
brand_encoder = LabelEncoder()
df['Brand'] = brand_encoder.fit_transform(df['Brand'])  # 將品牌轉換為數字編碼

# 特徵選擇
X = df[['Age', 'Gender', 'Item Purchased', 'Payment Method']]
y_amount = df['Purchase Amount (USD)'] > avg_purchase_amount  # 金額大於平均即為購買者
y_brand = df['Brand']  # 目標變數是品牌

# 標準化數據
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 訓練與測試數據分割
X_train, X_test, y_train_amount, y_test_amount = train_test_split(X_scaled, y_amount, test_size=0.2, random_state=42)
X_train, X_test, y_train_brand, y_test_brand = train_test_split(X_scaled, y_brand, test_size=0.2, random_state=42)

# 使用隨機森林模型進行購買金額預測
model_amount = RandomForestClassifier(n_estimators=100, random_state=42)
model_amount.fit(X_train, y_train_amount)

# 使用隨機森林模型進行品牌預測
model_brand = RandomForestClassifier(n_estimators=100, random_state=42)
model_brand.fit(X_train, y_train_brand)

# 根據自己填入的年齡與性別預測是否可能購買高於平均金額
input_age = st.number_input("Age", min_value=df['Age'].min(), max_value=df['Age'].max())
input_gender = st.selectbox("Gender", ['Male', 'Female'])
input_gender = le.transform([input_gender])[0]
input_item = st.selectbox("Item", item_encoder.inverse_transform(df['Item Purchased'].unique()))
input_payment_method = st.selectbox("Payment", payment_encoder.inverse_transform(df['Payment Method'].unique()))

# 預測購買金額
input_data = pd.DataFrame(
    [[input_age, input_gender, item_encoder.transform([input_item])[0], payment_encoder.transform([input_payment_method])[0]]],
    columns=['Age', 'Gender', 'Item Purchased', 'Payment Method']  # 確保欄位名稱與原始資料一致
)
input_data_scaled = scaler.transform(input_data)  # 無需報錯，特徵名稱已對齊
prediction_amount = model_amount.predict(input_data_scaled)[0]

if prediction_amount:
    st.markdown("**<span style='font-size: 30px;'>Above</span> the average amount.**", unsafe_allow_html=True)
else:
    st.markdown("**<span style='font-size: 30px;'>Under</span> the average amount.**", unsafe_allow_html=True)

# 預測品牌
prediction_brand = model_brand.predict(input_data_scaled)[0]
predicted_brand = brand_encoder.inverse_transform([prediction_brand])[0]
st.markdown(f"Likely to purchase <span style='font-size: 25px; font-weight: bold;'>{predicted_brand}</span>", unsafe_allow_html=True)



