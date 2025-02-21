import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import nltk
from nltk.corpus import twitter_samples
import re
from textblob import TextBlob
from wordcloud import WordCloud
import random
import datetime

# Download dataset if not available
nltk.download("twitter_samples")

# Load Twitter dataset from nltk
tweets = twitter_samples.strings("tweets.20150430-223406.json")

# Function to clean tweets
def clean_tweet(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r"@\w+", "", text)  # Remove mentions
    text = re.sub(r"#\w+", "", text)  # Remove hashtags
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Keep only letters and spaces
    return text.strip()

# Function to extract hashtags
def extract_hashtags(text):
    return re.findall(r"#\w+", text)

# Function to analyze sentiment
def analyze_sentiment(text):
    analysis = TextBlob(text).sentiment.polarity
    return "Positive" if analysis > 0 else "Negative" if analysis < 0 else "Neutral"

# Process tweets
tweet_data = []
hashtag_counts = {}

for tweet in tweets[:500]:  # Limit to 500 tweets for efficiency
    cleaned_text = clean_tweet(tweet)
    sentiment = analyze_sentiment(cleaned_text)
    hashtags = extract_hashtags(tweet)

    for tag in hashtags:
        hashtag_counts[tag] = hashtag_counts.get(tag, 0) + 1

    # Simulated engagement metrics
    likes = random.randint(10, 5000)
    retweets = random.randint(5, 2000)
    timestamp = datetime.datetime.now() - datetime.timedelta(minutes=random.randint(1, 1000))

    tweet_data.append({
        "Tweet": cleaned_text,
        "Sentiment": sentiment,
        "Likes": likes,
        "Retweets": retweets,
        "Timestamp": timestamp
    })

# Convert to DataFrame
df = pd.DataFrame(tweet_data)

# Streamlit Sidebar
st.sidebar.title("🔍 Filter Options")
sentiment_filter = st.sidebar.multiselect("Select Sentiment", ["Positive", "Negative", "Neutral"], default=["Positive", "Negative", "Neutral"])

# Filter Data
filtered_df = df[df["Sentiment"].isin(sentiment_filter)]

# Main Dashboard
st.title("🐦 Twitter Sentiment Analysis Dashboard")

# Sentiment Distribution (Pie Chart)
st.subheader("📊 Sentiment Distribution")
sentiment_counts = filtered_df["Sentiment"].value_counts().reset_index()
fig_pie = px.pie(sentiment_counts, names="Sentiment", values="count", color="Sentiment",
                 color_discrete_map={"Positive": "#2ECC71", "Negative": "#E74C3C", "Neutral": "#95A5A6"},
                 title="Sentiment Analysis of Tweets",
                 hole=0.4)  # Donut chart for better readability
st.plotly_chart(fig_pie)

# Engagement Metrics (Bar Chart)
st.subheader("📈 Engagement Insights")
engagement_fig = px.bar(filtered_df, x="Sentiment", y=["Likes", "Retweets"], barmode="group",
                        title="Likes & Retweets per Sentiment", color="Sentiment",
                        color_discrete_map={"Positive": "#2ECC71", "Negative": "#E74C3C", "Neutral": "#95A5A6"},
                        text_auto=True)
st.plotly_chart(engagement_fig)

# Word Cloud for Hashtags (Improved Aesthetics)
st.subheader("☁️ Trending Hashtags Word Cloud")
if hashtag_counts:
    wordcloud = WordCloud(width=800, height=400, background_color="black",
                          colormap="coolwarm", contour_color="white",
                          contour_width=2, max_words=100).generate_from_frequencies(hashtag_counts)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)
else:
    st.warning("No hashtags found.")

# Sentiment Over Time (Smoothed Line Chart)
st.subheader("⏳ Sentiment Trend Over Time")
time_df = filtered_df.groupby([pd.Grouper(key="Timestamp", freq="H"), "Sentiment"]).size().reset_index(name="Count")
fig_time = px.line(time_df, x="Timestamp", y="Count", color="Sentiment",
                   title="Sentiment Trend Over Time",
                   color_discrete_map={"Positive": "#2ECC71", "Negative": "#E74C3C", "Neutral": "#95A5A6"},
                   markers=True,  # Adds data points for better readability
                   line_shape="spline")  # Smooth curves instead of straight lines
st.plotly_chart(fig_time)

# Top Positive & Negative Tweets
st.subheader("🏆 Top Positive & Negative Tweets")
col1, col2 = st.columns(2)

with col1:
    st.subheader("🌟 Most Liked Positive Tweet")
    top_positive = filtered_df[filtered_df["Sentiment"] == "Positive"].nlargest(1, "Likes")
    st.success(top_positive["Tweet"].values[0] if not top_positive.empty else "No positive tweets found.")

with col2:
    st.subheader("💔 Most Liked Negative Tweet")
    top_negative = filtered_df[filtered_df["Sentiment"] == "Negative"].nlargest(1, "Likes")
    st.error(top_negative["Tweet"].values[0] if not top_negative.empty else "No negative tweets found.")

# Display Data
st.subheader("📜 Tweets Data")
st.dataframe(filtered_df.style.set_properties(**{'background-color': '#1E1E1E', 'color': 'white'}))

# Download CSV Button
st.subheader("⬇️ Download Data")
csv = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button("Download Data as CSV", data=csv, file_name="tweets_data.csv", mime="text/csv")
