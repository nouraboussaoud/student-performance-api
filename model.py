import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define features globally
FEATURES = ['Age', 'StudyTimeWeekly', 'Absences', 'GPA', 'ParentalEducation', 'ParentalSupport']

# Define feature ranges from training data
FEATURE_RANGES = {
    'Age': (15, 18),
    'StudyTimeWeekly': (0, 19.98),
    'Absences': (0, 30),  # Assumed max, adjust based on data
    'GPA': (0, 4),
    'ParentalEducation': (0, 4),
    'ParentalSupport': (0, 4)
}

def load_data(data_path='data/Student_performance_data _.csv'):
    try:
        df = pd.read_csv(data_path)
        logging.info(f"Data loaded. Shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def preprocess_data(df):
    try:
        # Check for missing values
        if df.isnull().sum().any():
            logging.warning("Missing values detected")
            logging.warning(df.isnull().sum())
        
        # Check value distributions
        logging.info("\nValue distributions:")
        logging.info(df.describe())
        
        # Check categorical value counts
        categorical_cols = ['ParentalEducation', 'ParentalSupport']
        for col in categorical_cols:
            if col in df.columns:
                logging.info(f"\n{col} value counts:")
                logging.info(df[col].value_counts())
        
        X = df[FEATURES]
        y = df['GradeClass']
        
        logging.info("\nGradeClass distribution:")
        logging.info(y.value_counts())
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, y, scaler, FEATURES
    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        raise

def train_gradient_boosting(X, y, model_path='models/gradient_boosting_model.pkl'):
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Compute sample weights to handle class imbalance
        class_counts = y_train.value_counts()
        total_samples = len(y_train)
        sample_weights = np.zeros(len(y_train))
        for cls in class_counts.index:
            weight = total_samples / (len(class_counts) * class_counts[cls])
            sample_weights[y_train == cls] = weight
        logging.info("Sample weights computed for classes: %s", class_counts.index.tolist())
        
        # Simplified GridSearchCV
        param_grid = {
            'n_estimators': [100],
            'learning_rate': [0.1],
            'max_depth': [3]
        }
        base_model = GradientBoostingClassifier(random_state=42)
        grid_search = GridSearchCV(base_model, param_grid, cv=3, scoring='accuracy', n_jobs=1)
        grid_search.fit(X_train, y_train, sample_weight=sample_weights)
        
        # Use the best model
        model = grid_search.best_estimator_
        logging.info(f"Best parameters: {grid_search.best_params_}")
        
        # Evaluate
        y_pred = model.predict(X_test)
        logging.info("Classification Report:")
        logging.info(classification_report(y_test, y_pred))
        logging.info("\nConfusion Matrix:")
        logging.info(confusion_matrix(y_test, y_pred))
        
        joblib.dump(model, model_path)
        logging.info(f"GradientBoosting model saved to {model_path}")
        return model, X_test, y_test
    except Exception as e:
        logging.error(f"Error training Gradient Boosting: {e}")
        raise

def train_kmeans(X, cluster_features, n_clusters=4, kmeans_path='models/kmeans_model.pkl', kmeans_scaler_path='models/kmeans_scaler.pkl'):
    try:
        feature_indices = [FEATURES.index(f) for f in cluster_features]
        X_cluster = X[:, feature_indices]
        
        # Use a dedicated scaler for clustering features
        scaler = StandardScaler()
        X_cluster_scaled = scaler.fit_transform(X_cluster)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(X_cluster_scaled)
        
        # Save both the model and the scaler
        joblib.dump(kmeans, kmeans_path)
        joblib.dump(scaler, kmeans_scaler_path)
        
        # Print centroids for debugging (scaled and unscaled)
        logging.info(f"KMeans model saved to {kmeans_path}")
        logging.info(f"KMeans scaler saved to {kmeans_scaler_path}")
        logging.info("KMeans centroids (scaled, [GPA, StudyTimeWeekly, Absences]):")
        for i, centroid in enumerate(kmeans.cluster_centers_):
            logging.info(f"Cluster {i}: {centroid}")
        logging.info("KMeans centroids (unscaled, [GPA, StudyTimeWeekly, Absences]):")
        unscaled_centroids = scaler.inverse_transform(kmeans.cluster_centers_)
        for i, centroid in enumerate(unscaled_centroids):
            logging.info(f"Cluster {i}: {centroid}")
        
        return kmeans
    except Exception as e:
        logging.error(f"Error training KMeans: {e}")
        raise

def predict_student_outcome(student_data, rf_model, gb_model, kmeans, scaler, cluster_features=None, model_type='gradient_boosting'):
    try:
        student_array = np.array(student_data)
        if student_array.ndim == 1:
            student_array = student_array.reshape(1, -1)
        elif student_array.ndim > 2:
            student_array = student_array.reshape(student_array.shape[0], -1)
        
        # Clip input features to training data ranges
        student_array_clipped = student_array.copy()
        for i, feature in enumerate(FEATURES):
            min_val, max_val = FEATURE_RANGES[feature]
            student_array_clipped[:, i] = np.clip(student_array[:, i], min_val, max_val)
        
        # Convert to DataFrame to preserve feature names
        student_df = pd.DataFrame(student_array_clipped, columns=FEATURES)
        
        # Debug: Print raw and clipped input data
        logging.info(f"Raw input data: {student_array}")
        logging.info(f"Clipped input data: {student_array_clipped}")
        
        # Apply scaling for Gradient Boosting
        scaled_data = scaler.transform(student_df)
        logging.info(f"Scaled input data: {scaled_data}")
        
        # Predict grade and success probability
        if model_type == 'gradient_boosting':
            predicted_grade = gb_model.predict(scaled_data)[0]
            probs = gb_model.predict_proba(scaled_data)[0]
            success_prob = probs[0] + probs[1]  # Sum of probabilities for GradeClass=0 (A) and GradeClass=1 (B)
            logging.info(f"Predicted grade: {predicted_grade}")
            logging.info(f"Probabilities (all classes): {probs}")
            logging.info(f"Success probability (A or B): {success_prob}")
        else:
            raise ValueError(f"Type de modèle non reconnu: {model_type}")
        
        # Calculate required study time
        current_study_time = student_array_clipped[0][FEATURES.index('StudyTimeWeekly')]
        required_study_time = current_study_time
        if success_prob < 0.7:
            required_study_time = min(current_study_time + 5, 40)
        
        # Determine risk level
        if success_prob >= 0.8:
            risk_level = "Faible"
        elif success_prob >= 0.5:
            risk_level = "Moyen"
        else:
            risk_level = "Élevé"
        
        # Generate recommendation and intervention
        recommendation = "Maintenir les efforts actuels" if success_prob >= 0.8 else "Augmenter le temps d'étude et demander un soutien supplémentaire"
        intervention = "Aucune" if success_prob >= 0.8 else "Former une équipe avec des étudiants performants"
        
        # Predict cluster
        cluster_info = {}
        if kmeans is not None and cluster_features is not None:
            try:
                # Load the kmeans scaler
                kmeans_scaler = joblib.load('models/kmeans_scaler.pkl')
                
                # Extract clustering features
                cluster_data = student_df[cluster_features]
                
                # Apply the kmeans scaler
                cluster_data_scaled = kmeans_scaler.transform(cluster_data)
                
                # Debug: Print scaled cluster data
                logging.info(f"Scaled cluster data for prediction: {cluster_data_scaled}")
                
                cluster = kmeans.predict(cluster_data_scaled)[0]
                logging.info(f"Predicted cluster: {cluster}")
                
                # Define cluster names and descriptions (to be adjusted based on centroids)
                cluster_names = {
                    0: "Étudiants performants avec assiduité élevée",
                    1: "Étudiants moyens avec assiduité variable",
                    2: "Étudiants à risque avec assiduité faible",
                    3: "Étudiants à potentiel d'amélioration"
                }
                
                cluster_descriptions = {
                    0: "GPA élevé, temps d'étude important, peu d'absences",
                    1: "GPA moyen, temps d'étude moyen, absences variables",
                    2: "GPA faible, temps d'étude faible, absences fréquentes",
                    3: "GPA moyen à faible, temps d'étude variable, absences modérées"
                }
                
                cluster_info = {
                    "cluster_id": int(cluster),
                    "cluster_name": cluster_names.get(cluster, f"Cluster inconnu {cluster}"),
                    "description": cluster_descriptions.get(cluster, "Description non disponible")
                }
            except Exception as e:
                import traceback
                logging.error(f"Erreur lors de la prédiction du cluster: {e}")
                logging.error(traceback.format_exc())
                cluster_info = {
                    "cluster_id": -1,
                    "cluster_name": "Erreur de clustering",
                    "description": f"Erreur: {str(e)}"
                }
        
        return {
            "predicted_grade": int(predicted_grade),
            "success_probability": float(success_prob),
            "required_study_time": float(required_study_time),
            "risk_level": risk_level,
            "recommendation": recommendation,
            "intervention": intervention,
            "cluster_info": cluster_info
        }
    except Exception as e:
        logging.error(f"Error predicting outcome: {e}")
        raise

def load_gradient_boosting(model_path='models/gradient_boosting_model.pkl'):
    try:
        return joblib.load(model_path)
    except Exception as e:
        logging.error(f"Error loading Gradient Boosting model: {e}")
        raise

def load_kmeans(kmeans_path='models/kmeans_model.pkl'):
    try:
        return joblib.load(kmeans_path)
    except Exception as e:
        logging.error(f"Error loading KMeans model: {e}")
        raise

def load_scaler(scaler_path='models/scaler.pkl'):
    try:
        return joblib.load(scaler_path)
    except Exception as e:
        logging.error(f"Error loading scaler: {e}")
        raise

if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    try:
        df = load_data()
        X_scaled, y, scaler, features = preprocess_data(df)
        joblib.dump(scaler, 'models/scaler.pkl')
        logging.info("Scaler saved")
        gb_model, _, _ = train_gradient_boosting(X_scaled, y)
        kmeans = train_kmeans(X_scaled, cluster_features=['GPA', 'StudyTimeWeekly', 'Absences'])
    except Exception as e:
        logging.error(f"Error in main: {e}")