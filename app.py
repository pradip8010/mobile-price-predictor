import streamlit as st

st.set_page_config(page_title="📱 Mobile Price Range Predictor", layout="wide")

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

@st.cache_data
def load_data():
    return pd.read_csv("mobile_price_prediction_5000.csv")

df = load_data()

# 🎯 Create price range classes using quartiles
df['Price_Range'] = pd.qcut(df['Price'], q=4, labels=["Low", "Mid", "High", "Premium"])

# ⚖️ Balance dataset
min_count = df['Price_Range'].value_counts().min()
df_balanced = pd.concat([
    df[df['Price_Range'] == label].sample(min_count, random_state=42)
    for label in df['Price_Range'].unique()
], ignore_index=True)

# 🧹 Drop strong leakage columns
df_balanced = df_balanced.drop(columns=['Model', 'Price', 'Storage'])

# 🔠 Encode categorical columns
brand_encoder = LabelEncoder()
df_balanced['Brand'] = brand_encoder.fit_transform(df_balanced['Brand'])
df_balanced['5G'] = LabelEncoder().fit_transform(df_balanced['5G'])

# 🧪 Features and Target
X = df_balanced.drop(columns=['Price_Range'])
y = df_balanced['Price_Range']

# 🔀 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ⚖️ Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 🤖 Model Training
model = LogisticRegression(max_iter=1000, random_state=42, multi_class="multinomial")
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

# 🎯 App Layout
st.markdown("<h1 style='text-align: center;'>📱 Mobile Price Range Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predict the <b>Price Range</b> of a mobile phone based on its features.</p>", unsafe_allow_html=True)
st.markdown("---")

# 🔢 Dataset Preview & Metrics
left_col, right_col = st.columns(2)

with left_col:
    st.markdown("### 📦 Dataset Sample")
    st.dataframe(df_balanced.head(10), height=300)

with right_col:
    st.markdown("### ✅ Model Evaluation")
    st.metric(label="🎯 Accuracy", value=f"{accuracy * 100:.2f}%")

    st.markdown("#### 🔁 Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", xticklabels=model.classes_, yticklabels=model.classes_, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

# 📌 Feature Importance
st.markdown("#### 📊 Feature Importance")
coef_df = pd.DataFrame(model.coef_, columns=X.columns, index=model.classes_).T
coef_df["Max_Influence"] = coef_df.abs().max(axis=1)
coef_sorted = coef_df.sort_values("Max_Influence", ascending=False).drop(columns="Max_Influence")

fig2, ax2 = plt.subplots(figsize=(8, 5))
coef_sorted.plot(kind='barh', ax=ax2)
plt.title("Feature Influence Across Price Ranges")
plt.tight_layout()
st.pyplot(fig2)

# 📈 Classification Report
st.markdown("### 📈 Classification Report")
st.code(classification_report(y_test, y_pred), language="text")

st.markdown("---")

# 🔍 Prediction Form
st.markdown("### 🤖 Try Your Own Prediction")

brand_names = brand_encoder.classes_

col1, col2, col3 = st.columns(3)

with col1:
    brand_input = st.selectbox("📛 Brand", brand_names)
    ram = st.slider("🧠 RAM (GB)", 2, 12, 6)

with col2:
    battery = st.slider("🔋 Battery (mAh)", 2000, 6000, 4000)
    screen = st.slider("📐 Screen Size (inches)", 4.0, 7.0, 6.5)

with col3:
    front = st.slider("🤳 Front Camera (MP)", 2, 64, 16)
    back = st.slider("📸 Rear Camera (MP)", 2, 108, 48)
    support_5g = st.radio("📶 5G Support", [0, 1], format_func=lambda x: "Yes" if x else "No")

# ✨ Prepare Input
brand_encoded = brand_encoder.transform([brand_input])[0]
input_data = np.array([[brand_encoded, ram, battery, screen, back, front, support_5g]])
input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)[0]

# 🎉 Result
st.markdown("---")
st.success(f"🔮 The phone is predicted to be in the **{prediction}** price range.")
st.markdown("<hr style='border:1px dashed #aaa'><p style='text-align:center'>✨ Built with ❤️ using Streamlit</p>", unsafe_allow_html=True)
