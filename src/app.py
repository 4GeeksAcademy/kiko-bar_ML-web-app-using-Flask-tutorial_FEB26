from flask import Flask, request, render_template
import pandas as pd
import joblib, os

app = Flask(__name__)

# 1. Get the directory that this app.py file is in (the 'src' folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Go UP one level and then into the 'models' folder
# This works on your computer AND on Render!
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'career_recommender.pkl')
VECTOR_COLS_PATH = os.path.join(BASE_DIR, '..', 'models', 'vector_cols.pkl')
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'processed', 'adult-census-income-cleaned.csv')

# 3. Load your files using these absolute paths
model = joblib.load(MODEL_PATH)
vector_cols = joblib.load(VECTOR_COLS_PATH)
df_cleaned = pd.read_csv(DATA_PATH)

@app.route('/')
def index():
    # This will show the input form (HTML)
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    # 1. Get data from the web form
    user_age = int(request.form.get('age'))
    user_education = request.form.get('education')
    user_hours = float(request.form.get('hours_per_week'))

    # 2. Vectorize the user input
    user_input_dict = {col: 0 for col in vector_cols}
    # Set the '1' for the selected education category
    # Example: if user picked Bachelors, 'education_Bachelors' becomes 1
    edu_col = f"education_{user_education}"
    if edu_col in user_input_dict:
        user_input_dict[edu_col] = 1
    # Convert dictionary to a single row (vector)
    user_vector = pd.DataFrame([user_input_dict])
    # 3. model.kneighbors() to find the match
    distances, indices = model.kneighbors(user_vector)
    recommendations = df_cleaned.iloc[indices[0]]

    # Select only the most interesting columns to display
    display_cols = ['age', 'education', 'occupation', 'hours.per.week', 'income']
    recommendations_to_show = recommendations[display_cols]

    # Pass THIS smaller version to the template
    table_html = recommendations_to_show.to_html(classes='data', index=False)
    # table_html = recommendations_to_show.to_dict(orient='records') --> another way to do it but required manually loop in the html.

    # 4. Return the result to a new page
    # this is for test --> render_template('recommendations.html', recommendations=recommendations.to_dict(orient='records'))
    return render_template('recommendations.html', table_html= table_html)

if __name__ == '__main__':
    app.run(debug=True)