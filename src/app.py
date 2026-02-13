from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load the "Brain" and the Data
model = joblib.load('./models/career_recommender.pkl')
vector_cols = joblib.load('./models/vector_cols.pkl')
df_cleaned = pd.read_csv('./data/processed/adult-census-income-cleaned.csv')

@app.route('/')
def index():
    # This will show your input form (HTML)
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    # 1. Get data from the web form
    # 2. Vectorize the user input
    # 3. model.kneighbors() to find the match
    # 4. Return the result to a new page
    return "Recommendation results will go here!"

if __name__ == '__main__':
    app.run(debug=True)