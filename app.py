import pandas as pd
import re
import string
from flask import Flask, request, jsonify, render_template
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import psycopg2

app = Flask(__name__)

# Database connection
def get_db_connection():
    conn = psycopg2.connect(
        dbname='newsdetectiondb',
        user='tanaya',
        password='cQsflHxOnfKr7Un7hCW7ZJVViGlqU1Sq',
        host='dpg-cp4sfj21hbls73f5ven0-a.singapore-postgres.render.com',
        port='5432'
    )
    return conn

# Load datasets from the database
def load_data_from_db():
    conn = get_db_connection()
    query_fake = "SELECT text, 0 AS class FROM Fake_News"
    query_true = "SELECT text, 1 AS class FROM True_News"

    data_fake = pd.read_sql(query_fake, conn)
    data_true = pd.read_sql(query_true, conn)
    conn.close()
    
    return pd.concat([data_fake, data_true]).sample(frac=1).reset_index(drop=True)

data = load_data_from_db()

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

# Train models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(random_state=0)
}

for model_name, model in models.items():
    model.fit(xv_train, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        title = request.form.get('title', '')
        title_processed = wordopt(title)
        
        # Check the input title against the database
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT title FROM True_News WHERE title = %s", (title,))
        db_result = cur.fetchone()
        conn.close()
        
        if db_result:
            db_verdict = "True News"
        else:
            db_verdict = "Fake News"

        return render_template('index.html', result=db_verdict)

if __name__ == '__main__':
    app.run(debug=True)
