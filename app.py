
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from model import load_data, preprocess_data, train_gradient_boosting, train_kmeans, predict_student_outcome, load_gradient_boosting, load_kmeans, load_scaler
import pandas as pd
import json
import joblib
import os
import sys

# Try to load .env, but continue if it fails
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Environment variables loaded from .env file.")
except Exception as e:
    print(f"Warning: Could not load .env file: {e}")
    print("Continuing with default values...")

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'fallback-secret-key')

DATA_PATH = 'data/Student_performance_data _.csv'
GB_MODEL_PATH = 'models/gradient_boosting_model.pkl'
KMEANS_PATH = 'models/kmeans_model.pkl'
SCALER_PATH = 'models/scaler.pkl'

try:
    df = load_data(DATA_PATH)
    X_scaled, y, scaler, features = preprocess_data(df)
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    if os.path.exists(GB_MODEL_PATH) and os.path.exists(KMEANS_PATH):
        gb_model = load_gradient_boosting(GB_MODEL_PATH)
        kmeans = load_kmeans(KMEANS_PATH)
    else:
        gb_model, _, _ = train_gradient_boosting(X_scaled, y, GB_MODEL_PATH)
        kmeans = train_kmeans(X_scaled, cluster_features=['GPA', 'StudyTimeWeekly', 'Absences'], n_clusters=4, kmeans_path=KMEANS_PATH)
except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure 'data/Student_performance_data.csv' exists.")
    sys.exit(1)
except Exception as e:
    print(f"Error loading models: {e}")
    sys.exit(1)

@app.route('/')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', features=features)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        try:
            with open('users_db.json', 'r') as f:
                users = json.load(f)
            if username in users and users[username] == password:
                session['username'] = username
                return jsonify({'success': True})
            return jsonify({'success': False, 'error': 'Invalid credentials'})
        except FileNotFoundError:
            return jsonify({'success': False, 'error': 'User database not found'})
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Afficher toutes les données reçues pour le débogage
        print("Données reçues du formulaire:", request.form)
        
        student_data = []
        for feature in features:
            # Vérifier si la clé existe dans le formulaire
            if feature not in request.form:
                return jsonify({'error': f'Champ manquant: {feature}'})
            
            # Récupérer la valeur et vérifier qu'elle n'est pas vide
            value = request.form[feature].strip()
            if not value:
                return jsonify({'error': f'Valeur manquante pour {feature}'})
            
            # Remplacer la virgule par un point pour les nombres décimaux
            value = value.replace(',', '.')
            
            try:
                # Convertir en nombre
                value = float(value)
            except ValueError:
                return jsonify({'error': f'La valeur "{request.form[feature]}" pour {feature} n\'est pas un nombre valide'})
            
            # Valider les plages
            if feature == 'Age' and (value < 10 or value > 25):
                return jsonify({'error': f'L\'âge doit être entre 10 et 25 ans'})
            if feature == 'GPA' and (value < 0 or value > 4):
                return jsonify({'error': f'Le GPA doit être entre 0 et 4'})
            if feature == 'Absences' and value < 0:
                return jsonify({'error': f'Le nombre d\'absences ne peut pas être négatif'})
            if feature == 'StudyTimeWeekly' and (value < 0 or value > 40):
                return jsonify({'error': f'Le temps d\'étude doit être compris entre 0 et 40 heures'})
            
            student_data.append(value)
        
        # Afficher les données après traitement pour débogage
        print("Données traitées:", student_data)
        
        result = predict_student_outcome([student_data], None, gb_model, kmeans, scaler, 
                                        cluster_features=['GPA', 'StudyTimeWeekly', 'Absences'], 
                                        model_type='gradient_boosting')
        return jsonify(result)
    except Exception as e:
        import traceback
        print("Erreur détaillée:", traceback.format_exc())
        return jsonify({'error': f'Erreur: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)
