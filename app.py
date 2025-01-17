
# Import libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import joblib
import os

# Streamlit app title
st.title("Social Media Sentiment Analysis")

# Load data with updated caching
@st.cache_data
def load_data(filepath):
    return pd.read_csv(filepath)

data_path = "E:\\synthetic social media data.csv"
data = load_data(data_path)

# Display dataset preview
st.header("Uploaded Dataset")
st.write(data.head())

# Preprocessing functions
@st.cache_data
def preprocess_data(data):
    features = data[['Post Content']]
    labels_classification = data[['Sentiment Label']]
    labels_regression = data[['Number of Likes', 'Number of Shares', 'Number of Comments', 'User Follower Count']]
    return features, labels_classification, labels_regression

features, labels_classification, labels_regression = preprocess_data(data)

# Vectorize text data
@st.cache_resource
def vectorize_text(features):
    vectorizer = TfidfVectorizer(max_features=500)  # Reduced max_features for speed
    X = vectorizer.fit_transform(features['Post Content']).toarray()
    return vectorizer, X

vectorizer, X = vectorize_text(features)

# Split data
X_train, X_test, y_train_class, y_test_class = train_test_split(
    X, labels_classification, test_size=0.2, random_state=42
)
_, _, y_train_reg, y_test_reg = train_test_split(
    X, labels_regression, test_size=0.2, random_state=42
)

# Train or Load Models
@st.cache_resource
def train_or_load_models(X_train, y_train_class, y_train_reg):
    if os.path.exists('classification_model.pkl') and os.path.exists('regression_model.pkl'):
        clf = joblib.load('classification_model.pkl')
        reg = joblib.load('regression_model.pkl')
    else:
        clf = RandomForestClassifier(random_state=42, n_estimators=50)  # Reduced estimators
        clf.fit(X_train, y_train_class.values.ravel())

        reg = MultiOutputRegressor(RandomForestRegressor(random_state=42, n_estimators=50))
        reg.fit(X_train, y_train_reg)

        # Save the models
        joblib.dump(clf, 'classification_model.pkl')
        joblib.dump(reg, 'regression_model.pkl')
    return clf, reg

clf, reg = train_or_load_models(X_train, y_train_class, y_train_reg)

# Visualizations
st.header("Visualize Sentiment Analysis")
fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(data=data, x='Sentiment Label', palette='viridis', ax=ax)
plt.title("Sentiment Distribution")
st.pyplot(fig)

# User Input Section
st.header("Predict Sentiment from User Input")
user_input = st.text_area("Enter a post to analyze sentiment:")

if st.button("Analyze"):
    input_vectorized = vectorizer.transform([user_input])
    sentiment = clf.predict(input_vectorized)
    metrics = reg.predict(input_vectorized)

    st.write("**Predicted Sentiment Label:**", sentiment[0])
    st.write("**Predicted Number of Likes:**", int(metrics[0, 0]))
    st.write("**Predicted Number of Shares:**", int(metrics[0, 1]))
    st.write("**Predicted Number of Comments:**", int(metrics[0, 2]))
    st.write("**Predicted User Follower Count:**", int(metrics[0, 3]))

st.info("This application provides sentiment analysis and insights using optimized machine learning models.")
