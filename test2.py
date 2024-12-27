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
    return pd.read_csv('shopping_behavior_updated.csv')

# 加載數據
df = load_data()

# 行為模式與預測（簡單的機器學習示例）
st.title("Predict purchasing behavior")

#平均購買金額
avg_purchase_amount = df['Purchase Amount (USD)'].mean()
    
 # 邏輯回歸模型來預測購買金額
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
    
# 商品與支付方式的數字化
item_encoder = LabelEncoder()
df['Item Purchased'] = item_encoder.fit_transform(df['Item Purchased'])
payment_encoder = LabelEncoder()
df['Payment Method'] = payment_encoder.fit_transform(df['Payment Method'])
    
# 特徵選擇
X = df[['Age', 'Gender', 'Item Purchased', 'Payment Method']]
y = df['Purchase Amount (USD)'] > avg_purchase_amount  # 金額大於平均即為購買者
    
# 標準化數據
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
    
# 訓練與測試數據分割
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
# 使用隨機森林模型進行預測
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
    
    # 預測準確率
    #accuracy = model.score(X_test, y_test)
    #st.write(f"隨機森林模型準確率: {accuracy:.2f}")
    
    # 交叉驗證評估模型
    #cross_val_scores = cross_val_score(model, X_scaled, y, cv=5)
    #st.write(f"交叉驗證平均準確率: {cross_val_scores.mean():.2f}")
    
    # 根據自己填入的年齡與性別預測是否可能購買高於平均金額
input_age = st.number_input("Age", min_value=df['Age'].min(), max_value=df['Age'].max())
input_gender = st.selectbox("Gender", ['Male', 'Female'])
input_gender = le.transform([input_gender])[0]
input_item = st.selectbox("Item", item_encoder.inverse_transform(df['Item Purchased'].unique()))
input_payment_method = st.selectbox("Payment", payment_encoder.inverse_transform(df['Payment Method'].unique()))
    
input_data = [[input_age, input_gender, item_encoder.transform([input_item])[0], payment_encoder.transform([input_payment_method])[0]]]
input_data_scaled = scaler.transform(input_data)
    
prediction = model.predict(input_data_scaled)[0]
if prediction:
    st.write(f"Prediction: This consumer is likely to make a purchase above the average amount.")
else:
    st.write(f"Prediction: This consumer is likely to make a purchase under the average amount.")
