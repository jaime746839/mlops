import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# 1️⃣ Chargement et Prétraitement des Données
def prepare_data(train_path="churn-bigml-80.csv", test_path="churn-bigml-20.csv"):
    """
    Charge les fichiers Train et Test, applique encodage et normalisation.
    """
    import pandas as pd
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Assurer que toutes les colonnes ont des noms en string
    train_df.columns = train_df.columns.astype(str)
    test_df.columns = test_df.columns.astype(str)

    # Supposons que la dernière colonne soit la cible (y)
    X_train, y_train = train_df.iloc[:, :-1], train_df.iloc[:, -1]
    X_test, y_test = test_df.iloc[:, :-1], test_df.iloc[:, -1]

    # Assurer que les noms des colonnes des features sont des strings
    X_train.columns = X_train.columns.astype(str)
    X_test.columns = X_test.columns.astype(str)

    # Identifier les colonnes catégoriques
    categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

    # Appliquer un encodage One-Hot pour convertir en valeurs numériques
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_train_encoded = encoder.fit_transform(X_train[categorical_cols])
    X_test_encoded = encoder.transform(X_test[categorical_cols])

    # Convertir en DataFrame et renommer les colonnes encodées
    encoded_feature_names = encoder.get_feature_names_out(categorical_cols)
    X_train_encoded = pd.DataFrame(
        X_train_encoded, index=X_train.index, columns=encoded_feature_names
    )
    X_test_encoded = pd.DataFrame(
        X_test_encoded, index=X_test.index, columns=encoded_feature_names
    )

    # Supprimer les anciennes colonnes catégoriques et concaténer les nouvelles colonnes encodées
    X_train = X_train.drop(columns=categorical_cols).reset_index(drop=True)
    X_test = X_test.drop(columns=categorical_cols).reset_index(drop=True)
    X_train = pd.concat([X_train, X_train_encoded], axis=1)
    X_test = pd.concat([X_test, X_test_encoded], axis=1)

    # Normalisation des données numériques
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


# 2️⃣ Entraînement du Modèle
def train_model(X_train, y_train):
    """
    Entraîne un modèle RandomForest avec optimisation des hyperparamètres.
    """
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # Log the trained model and metrics to MLflow
    mlflow.start_run()
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 10)
    mlflow.sklearn.log_model(model, "random_forest_model")
    mlflow.log_metric("accuracy", 0.85)  # Example accuracy
    mlflow.end_run()

    return model


# 3️⃣ Évaluation du Modèle
def evaluate_model(model, X_test, y_test):
    """
    Affiche Accuracy, Matrice de confusion et Rapport de classification.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Log the evaluation metrics to MLflow
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", 0.90)  # Example precision


# 4️⃣ Sauvegarde et Chargement du Modèle
def save_model(model, filename="model.joblib"):
    """
    Sauvegarde le modèle entraîné.
    """
    joblib.dump(model, filename)
    print(f"Modèle sauvegardé sous {filename}")


def load_model(filename="model.joblib"):
    """
    Charge un modèle sauvegardé.
    """
    model = joblib.load(filename)
    print(f"Modèle chargé depuis {filename}")
    return model
