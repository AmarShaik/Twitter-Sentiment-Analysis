# Twitter Sentiment Analysis

This project, **Twitter Sentiment Analysis**, aims to analyze the sentiment of tweets using Natural Language Processing (NLP) techniques and machine learning. The analysis classifies tweets into positive, negative, or neutral sentiments based on their content. 

By leveraging the Sentiment140 dataset and integrating with the Twitter API, this project demonstrates how data-driven insights can be derived from social media platforms. 

The project involves multiple stages, including:
- Fetching datasets from Kaggle.
- Retrieving real-time tweets using the Twitter API.
- Cleaning and preprocessing textual data.
- Conducting sentiment analysis using the TextBlob library.
- Visualizing the sentiment distribution for deeper understanding.

This comprehensive workflow provides a scalable and practical approach for analyzing social media data for sentiment analysis tasks, which can be applied in industries like marketing, customer service, and public relations.

## Steps to Set Up the Project

### 1. Install Required Libraries
The first step is to install the necessary dependencies for this project. This can be done using pip:
```bash
pip install kaggle tweepy textblob pandas matplotlib seaborn nltk
```

### 2. Import the Dependencies
The following libraries are used in this project:
- **kaggle**: For downloading datasets from Kaggle.
- **tweepy**: To interact with the Twitter API and fetch tweets.
- **textblob**: For sentiment analysis.
- **pandas**: To manage and manipulate datasets.
- **matplotlib** and **seaborn**: For data visualization.
- **nltk**: For natural language preprocessing.

### 3. Set Up Kaggle API for Dataset
To access datasets from Kaggle:
1. Visit [Kaggle](https://www.kaggle.com/) and log in.
2. Navigate to your account settings and download the Kaggle API token (a JSON file named `kaggle.json`).
3. Place `kaggle.json` in the appropriate directory (e.g., `~/.kaggle/` on Linux or `%USERPROFILE%\.kaggle\` on Windows).
4. Ensure Kaggle CLI is authenticated:
   ```bash
   kaggle datasets list
   ```

### 4. Fetch and Load Dataset
Use the Kaggle API to download the Sentiment140 dataset:
```bash
kaggle datasets download -d kazanova/sentiment140
```
Extract the dataset and load it into a pandas DataFrame for analysis.

### 5. Fetch Tweets Using Twitter API
Authenticate with the Twitter API using Tweepy to collect tweets based on specific keywords or hashtags.

**Example Code:**
```python
import tweepy

# API keys
consumer_key = "YOUR_API_KEY"
consumer_secret = "YOUR_API_SECRET"
access_token = "YOUR_ACCESS_TOKEN"
access_token_secret = "YOUR_ACCESS_TOKEN_SECRET"

# Authenticate
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# Fetch tweets
tweets = api.search_tweets(q="keyword", lang="en", count=100)
for tweet in tweets:
    print(tweet.text)
```

### 6. Preprocess the Data
Clean the tweets by:
- Removing URLs, stopwords, punctuation, and special characters.
- Lowercasing the text.

**Example Code:**
```python
import re
from nltk.corpus import stopwords

# Preprocess text
def preprocess_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^A-Za-z ]+', '', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text

example_tweet = "I love this product! http://example.com"
cleaned_tweet = preprocess_text(example_tweet)
print(cleaned_tweet)
```

### 7. Perform Sentiment Analysis
Use the **TextBlob** library to classify sentiment.

**Example Code:**
```python
from textblob import TextBlob

text = "I love this product!"
sentiment = TextBlob(text).sentiment.polarity
print("Sentiment Score:", sentiment)
```

### 8. Visualize the Results
Use **matplotlib** or **seaborn** to visualize sentiment distributions.

**Example Code:**
```python
import matplotlib.pyplot as plt

# Example: Sentiment counts
sentiments = ['positive', 'negative', 'neutral']
counts = [50, 30, 20]
plt.bar(sentiments, counts)
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()
```

---

## Repository Structure
```
.
├── dataset/               # Contains the Sentiment140 dataset
├── scripts/               # Python scripts for fetching and analyzing tweets
├── results/               # Output plots and results
└── README.md              # Project instructions (this file)
```

## How to Run
1. Clone the repository:
   ```bash
   git clone <repository_url>
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your Twitter API keys in the script.
4. Run the scripts to fetch, preprocess, and analyze tweets.

## References
- [Twitter Developer Portal](https://developer.twitter.com/)
- [Sentiment140 Dataset on Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140)

---
