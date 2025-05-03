import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
import joblib
import os

def load_data(data_path='data/Student_performance_data _.csv'):
    df = pd.read_csv(data_path)
    print(f"✅ Data loaded. Shape: {df.shape}")
    return df

def preprocess_data(df):
    # Check for missing values
    if df.isnull().sum().any():
        print("Warning: Missing values detected")
        print(df.isnull().sum())
    
    # Check value distributions
    print("\nValue distributions:")
    print(df.describe())
    
    # Check categorical value counts
    categorical_cols = ['ParentalEducation', 'StudyTimeWeekly', 'ParentalSupport']
    for col in categorical_cols:
        if col in df.columns:
            print(f"\n{col} value counts:")
            print(df[col].value_counts())
    le = LabelEncoder()
    categorical_cols = ['ParentalEducation', 'StudyTimeWeekly', 'ParentalSupport']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])
    features = ['Age', 'StudyTimeWeekly', 'Absences', 'GPA', 'ParentalEducation', 'ParentalSupport']
    X = df[features]
    y = df['GradeClass']
    # In preprocess_data()
    print("\nGradeClass distribution:")
    print(y.value_counts())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler, features

def train_model(X, y, model_path='models/random_forest_model.pkl'):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    model = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring='accuracy'
    )
    model.fit(X_train, y_train)
    print("Best parameters:", model.best_params_)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    # Evaluate
    from sklearn.metrics import classification_report, confusion_matrix
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    joblib.dump(model, model_path)
    return model, X_test, y_test

def train_gradient_boosting(X, y, model_path='models/gradient_boosting_model.pkl'):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    print(f"✅ GradientBoosting model saved to {model_path}")
    return model, X_test, y_test

def train_kmeans(X, cluster_features, n_clusters=4, kmeans_path='models/kmeans_model.pkl'):
    feature_indices = [features.index(f) for f in cluster_features]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X[:, feature_indices])
    joblib.dump(kmeans, kmeans_path)
    print(f"✅ KMeans model saved to {kmeans_path}")
    return kmeans

def predict_student_outcome(student_data, rf_model, gb_model, kmeans, scaler, cluster_features=None, model_type='random_forest'):
    """
    Prédire la probabilité de réussite d'un étudiant et son cluster.
    
    Args:
        student_data (list): Liste contenant les caractéristiques d'un étudiant
        rf_model: Modèle Random Forest entraîné
        gb_model: Modèle Gradient Boosting entraîné
        kmeans: Modèle K-means entraîné
        scaler: StandardScaler pour normaliser les données
        cluster_features (list): Liste des noms des caractéristiques utilisées pour le clustering
        model_type (str): Type de modèle à utiliser ('random_forest', 'gradient_boosting', ou 'ensemble')
    
    Returns:
        dict: Dictionnaire contenant la probabilité de réussite et les informations de cluster
    """
    # Assurer que student_data est un tableau numpy 2D
    # Si c'est une liste de listes, convertir en numpy array
    student_array = np.array(student_data)
    
    # S'assurer que c'est un tableau 2D (échantillons, caractéristiques)
    if student_array.ndim == 1:
        student_array = student_array.reshape(1, -1)
    elif student_array.ndim > 2:
        # Si le tableau a plus de 2 dimensions, réduire à 2D
        # Cela pourrait être nécessaire si les données ont une structure plus complexe
        student_array = student_array.reshape(student_array.shape[0], -1)
    
    # Appliquer la mise à l'échelle
    scaled_data = scaler.transform(student_array)
    
    # Prédire la probabilité de réussite selon le modèle choisi
    if model_type == 'random_forest':
        success_prob = rf_model.predict_proba(scaled_data)[0][1]
    elif model_type == 'gradient_boosting':
        success_prob = gb_model.predict_proba(scaled_data)[0][1]
    elif model_type == 'ensemble':
        rf_prob = rf_model.predict_proba(scaled_data)[0][1]
        gb_prob = gb_model.predict_proba(scaled_data)[0][1]
        success_prob = (rf_prob + gb_prob) / 2
    else:
        raise ValueError(f"Type de modèle non reconnu: {model_type}")
    
    # Prédire le cluster
    cluster_info = {}
    if kmeans is not None and cluster_features is not None:
        # Extraire les caractéristiques pertinentes pour le clustering
        # En supposant que l'ordre des caractéristiques dans student_data correspond à celui utilisé lors de l'entraînement
        # Ici vous devriez adapter selon votre implémentation réelle
        
        try:
            # Option 1: Si vous avez un index ou des positions connues pour les caractéristiques de clustering
            # Par exemple, si GPA est à l'index 3, StudyTimeWeekly à l'index 1, Absences à l'index 2
            feature_indices = {'GPA': 3, 'StudyTimeWeekly': 1, 'Absences': 2}  # À ajuster selon votre structure
            cluster_data = np.array([[student_array[0][feature_indices[feature]] for feature in cluster_features]])
            
            # Option 2: Alternative si vous avez une façon différente d'accéder aux caractéristiques
            # Cette partie doit être adaptée à votre implémentation réelle
            
            # Prédire le cluster
            cluster = kmeans.predict(cluster_data)[0]
            
            # Générer les informations sur le cluster
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
                "cluster_name": cluster_names.get(cluster, f"Cluster {cluster}"),
                "description": cluster_descriptions.get(cluster, "Pas d'information disponible pour ce groupe")
            }
        except Exception as e:
            import traceback
            print(f"Erreur lors de la prédiction du cluster: {e}")
            print(traceback.format_exc())
            # En cas d'erreur, on continue sans informations de cluster
    
    return {
        "success_probability": float(success_prob),
        "cluster_info": cluster_info
    }

def load_model(model_path='models/random_forest_model.pkl'):
    return joblib.load(model_path)

def load_gradient_boosting(model_path='models/gradient_boosting_model.pkl'):
    return joblib.load(model_path)

def load_kmeans(kmeans_path='models/kmeans_model.pkl'):
    return joblib.load(kmeans_path)

def load_scaler(scaler_path='models/scaler.pkl'):
    return joblib.load(scaler_path)

if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    try:
        df = load_data()
        X_scaled, y, scaler, features = preprocess_data(df)
        joblib.dump(scaler, 'models/scaler.pkl')
        print("✅ Scaler saved")
        rf_model, X_test, y_test = train_model(X_scaled, y)
        gb_model, _, _ = train_gradient_boosting(X_scaled, y)
        kmeans = train_kmeans(X_scaled, cluster_features=['GPA', 'StudyTimeWeekly', 'Absences'])
    except Exception as e:
        print(f"Error: {e}")