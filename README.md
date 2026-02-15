# 👔 Census-Based Career Recommender

A Full-Stack Machine Learning web application that uses a **k-Nearest Neighbors (kNN)** model to find "career twins" from the US Census database.

   **Live Demo:** https://kiko-bar-ml-web-app-using-flask-tutorial.onrender.com

## 🛠️ How it Works
1. **The Model:** Trained using Scikit-Learn on the Adult Census Income dataset.
2. **The Logic:** Takes user input (age, education, hours per week) and calculates the mathematical distance to find the 5 most similar profiles in the database.
3. **The Filter:** Strictly limited to profiles earning **<=50K** to focus on entry-to-mid-level career benchmarking.
4. **The Stack:** Python (Flask), Pandas, Joblib, HTML/CSS, and Jinja2.

## 📁 Project Structure
- `/src`: Flask application logic and routing.
- `/models`: Serialized kNN model and vectorization files.
- `/templates`: Custom HTML results using Jinja2 loops.
- `/data/processed`: Cleaned census data for results retrieval.
