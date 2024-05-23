import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load datasets
data_fake = pd.read_csv('Fake.csv')
data_true = pd.read_csv('True.csv')

# Assign class labels
data_fake['class'] = 0
data_true['class'] = 1

# Merge datasets and drop unnecessary columns
data = pd.concat([data_fake, data_true]).drop(['title', 'subject', 'date'], axis=1).sample(frac=1).reset_index(drop=True)

# Check for NaN values in the 'text' column and fill them with an empty string
data['text'] = data['text'].fillna('')

# Text preprocessing function
def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r"\\W", " ", text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

data['text'] = data['text'].apply(wordopt)

# Split data into training and testing sets
x = data['text']
y = data['class']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# Vectorize the text data
vectorizer = TfidfVectorizer()
xv_train = vectorizer.fit_transform(x_train)
xv_test = vectorizer.transform(x_test)

# Train and evaluate models
def train_and_evaluate_model(model, model_name):
    model.fit(xv_train, y_train)
    predictions = model.predict(xv_test)
    print(f"{model_name} Accuracy:", model.score(xv_test, y_test))
    print(classification_report(y_test, predictions))

models = [
    (LogisticRegression(), "Logistic Regression"),
    (DecisionTreeClassifier(), "Decision Tree"),
    (RandomForestClassifier(random_state=0), "Random Forest")
]

for model, model_name in models:
    train_and_evaluate_model(model, model_name)

# Manual testing function
def manual_testing(news):
    news = pd.DataFrame({'text': [news]})
    news['text'] = news['text'].apply(wordopt)
    news_vectorized = vectorizer.transform(news['text'])

    results = {}
    for model, model_name in models:
        model.fit(xv_train, y_train)  # Ensure each model is fitted before predicting
        prediction = model.predict(news_vectorized)
        results[model_name] = "Fake News" if prediction[0] == 0 else "True News"
    
    for model_name, result in results.items():
        print(f"{model_name} Prediction: {result}")

# Input news for manual testing
news = input("Enter news text for manual testing: ")
manual_testing(news)
