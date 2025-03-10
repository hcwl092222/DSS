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
from sklearn.cluster import KMeans

# 更改工作目錄
current = os.path.dirname(os.path.abspath(__file__))
os.chdir(current)

@st.cache_data
def load_data():
    df = pd.read_csv(
        'shopping_behavior.csv',
        usecols=["Customer ID","Age", "Gender", "Item Purchased", "Purchase Amount (USD)","Frequency of Purchases",'Payment Method',"Brand"]
    )
    # 確保 Age 為數值類型
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df = df[(df['Age'] >= 14) & (df['Age'] <= 29)]
    df['Purchase Amount (USD)'] = pd.to_numeric(df['Purchase Amount (USD)'], errors='coerce')  # 確保購買金額為數值類型
    return df.set_index('Item Purchased') #將Item設為index


def calculate_item_age_distribution(df):
    # 計算購買人數
    item_age_distribution = df.groupby(['Item Purchased', 'Age']).size().reset_index(name='Count')
    return item_age_distribution

# 載入數據
df = load_data()

tab1,tab2,tab3,tab4,tab5= st.tabs(["Item","Price","Consumer Behavior Analysis","Popular Items and Brands by Age","Consumer Behavior Clustering"])

with tab1:
    st.title("Item purchased data")
    # 品項不重複(搜尋列)
    unique_items = df.index.unique().tolist()
    item = st.multiselect(
        "Choose item", unique_items, ["Sweater", "Jeans"]
    )

    if not item:
        st.error("Please choose item")
    else:
        data = df.loc[item]

        st.subheader("Table")
        st.dataframe(data.sort_index())

        # 計算分布
        distribution = calculate_item_age_distribution(df.reset_index())
        filtered_distribution = distribution[distribution['Item Purchased'].isin(item)]

        st.subheader("Chart")
    
        chart = (
            alt.Chart(filtered_distribution)
            .mark_bar()
            .encode(
                x=alt.X("Age:O", title="Age"),
                y=alt.Y("Count:Q", title="Number of Purchases"),
                color=alt.Color("Item Purchased:N", title="Item Purchased"),
                tooltip=["Item Purchased", "Age", "Count"]
            )
        )

        st.altair_chart(chart, use_container_width=True)



with tab2:
    # 箱形圖
    # box_plot = alt.Chart(df).mark_boxplot().encode(
    #     x='Age:O',
    #     y='Purchase Amount (USD):Q'
    # )
    # st.altair_chart(box_plot, use_container_width=True)
    # 為年齡創建唯一的顏色映射
    st.title("Price data")
    unique_ages = sorted(df['Age'].unique())
    color_mapping = {age: f'rgb({i * 50 % 255}, {(i * 75 + 50) % 255}, {(i * 100+ 100) % 255})' for i, age in enumerate(unique_ages)}

    # 在數據框中新增顏色列
    df['Color'] = df['Age'].map(color_mapping)

    # Box Plot
    st.header("Box Plot")
    fig = px.box(df, x="Age", y="Purchase Amount (USD)", color="Age", color_discrete_map=color_mapping)
    st.plotly_chart(fig, use_container_width=True)

    # 使用 create_distplot 繪製分佈圖
    # 將數據按年齡分組
    st.header("Displot")
    ages = sorted(df['Age'].unique()) 
    purchase_data = [df[df['Age'] == age]['Purchase Amount (USD)'].dropna().tolist() for age in ages]


    fig = ff.create_distplot(
        purchase_data,         
        [f"Age {age}" for age in ages], 
        bin_size=10         
    )

    # 設置標題
    fig.update_layout(
        xaxis_title="Purchase Amount (USD)",  
        yaxis_title="Density",  
    )
    st.plotly_chart(fig)

with tab3:
     # 基本統計分析
        st.title("Consumer Behavior Analysis")
        
        avg_purchase_amount = df['Purchase Amount (USD)'].mean()
        most_popular_item = (
            distribution.groupby('Item Purchased')['Count']
            .sum()
            .idxmax()
        )
        most_frequent_payment_method = df['Payment Method'].mode()[0]
        
        st.write(f"Average Purchase Amount: ${avg_purchase_amount:.2f}")
        st.write(f"The most popular item: {most_popular_item}")
        st.write(f"Most Common Payment Methods: {most_frequent_payment_method}")

        # 將 'Item Purchased' 轉回普通列
        item_counts = df.reset_index()['Item Purchased'].value_counts().head(10)

        # 繪製圖表
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))

        # 1. 商品購買分佈
        sns.barplot(x=item_counts.values, y=item_counts.index, ax=ax[0], palette='Spectral', hue=item_counts.index)
        ax[0].set_title("Top 10 Most Popular Items")
        ax[0].set_xlabel("Number of Purchases")
        ax[0].set_ylabel("Item Name")

        # 2. 支付方式分佈
        sns.countplot(x='Payment Method', data=df, ax=ax[1], palette='vlag', hue='Payment Method')
        ax[1].set_title("Payment Method Distribution")
        ax[1].set_ylabel("Count")

        # 在 Streamlit 中顯示圖表
        st.pyplot(fig)
with tab4: 
    st.header("Popular Items and Brands by Age")
    
    # 年齡篩選
    min_age = df['Age'].min()
    max_age = df['Age'].max()
    
    # 用戶選擇年齡範圍
    age_range = st.slider("Choose the age range:", min_age, max_age, (min_age, max_age))
    
    filtered_df = df[(df['Age'] >= age_range[0]) & (df['Age'] <= age_range[1])]
    
    st.write(f"The age range：{age_range[0]} - {age_range[1]}")
    st.write(f"The filtered data contains a total of {filtered_df.shape[0]} entries")
    
    if not filtered_df.empty:
        # 將索引重設為普通列（僅在這裡進行操作）
        filtered_df_reset = filtered_df.reset_index()

        # 分析篩選後的商品購買次數
        filtered_item_counts = filtered_df_reset['Item Purchased'].value_counts().head(5)
        
        st.write("Top 5 most popular items in the selected age range:")
        st.write(filtered_item_counts)

        # 分析篩選後的品牌購買次數
        if 'Brand' in filtered_df.columns:  # 確保有 Brand 欄位
            filtered_brand_counts = filtered_df_reset['Brand'].value_counts().head(5)
            st.write("Top 5 most popular brands in the selected age range:")
            st.write(filtered_brand_counts)
        else:
            st.write("Brand data not available.")
with tab5:
    st.header("Clustering of Consumers Based on Behavior")

    # 使用新的 frequency_mapping 進行映射
    frequency_mapping = {
        "Weekly": 1,            # 每週
        "Bi-Weekly": 2,         # 每兩週
        "Fortnightly": 2,       # 每兩週
        "Monthly": 4,           # 每月（大約四週）
        "Quarterly": 13,        # 每季（大約13週）
        "Every 3 Months": 13,   # 每季
        "Annually": 52          # 每年（大約52週）
    }

    # 映射 'Frequency of Purchases' 欄位
    df['Frequency of Purchases'] = df['Frequency of Purchases'].map(frequency_mapping)

    # 檢查是否成功映射
    #if df['Frequency of Purchases'].isnull().any():
        #st.error("Some values in 'Frequency of Purchases' could not be mapped. They will be replaced with '0'.")
        #df['Frequency of Purchases'] = df['Frequency of Purchases'].fillna(0)
    #else:
        #st.success("All values in 'Frequency of Purchases' successfully mapped.")

    # 選擇需要聚類的特徵
    clustering_features = df[['Age', 'Frequency of Purchases', 'Purchase Amount (USD)']]
    scaler = StandardScaler()
    clustering_features_scaled = scaler.fit_transform(clustering_features)

    # 執行 K-Means 聚類
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(clustering_features_scaled)

    # 聚類結果可視化
    #st.subheader("Clustering Results (3 groups)")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        x=df['Age'],
        y=df['Purchase Amount (USD)'],
        hue=df['Cluster'],
        palette='viridis',
        ax=ax
    )
    ax.set_title("Clustering Visualization")
    ax.set_xlabel("Age")
    ax.set_ylabel("Purchase Amount (USD)")
    st.pyplot(fig)

    # 聚類分布統計
    st.write("Cluster Distribution:")
    cluster_counts = df['Cluster'].value_counts()
    # 添加 Cluster 註解
    st.write("""
    - **Cluster 0**: Likely to have higher frequency of purchases but lower purchase amounts.
    - **Cluster 1**: Likely to have moderate purchase frequency and higher purchase amounts.
    - **Cluster 2**: Likely to have lower frequency of purchases with moderate to high purchase amounts.
    """)
    
    st.write(cluster_counts)





