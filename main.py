import argparse
from model_pipeline import (evaluate_model, prepare_data, save_model,
                            train_model, load_model)
import mlflow
import mlflow.sklearn
from elasticsearch import Elasticsearch

# Set up Elasticsearch client (updated host to 'elasticsearch')
es = Elasticsearch([{'scheme': 'http', 'host': 'localhost', 'port': 9200}])  # Or 'host.docker.internal'


# Function to log metrics to Elasticsearch (no doc_type needed anymore)
def log_to_elasticsearch(run_id, metrics):
    doc = {
        'run_id': run_id,
        'metrics': metrics
    }
    es.index(index='mlflow-metrics', body=doc)  # Removed doc_type argument

def main():
    parser = argparse.ArgumentParser(description="Pipeline de Machine Learning")
    parser.add_argument("--prepare", action="store_true", help="Préparer les données")
    parser.add_argument("--train", action="store_true", help="Entraîner le modèle")
    parser.add_argument("--validate", action="store_true", help="Évaluer le modèle")
    args = parser.parse_args()

    # Préparer les données une seule fois et les réutiliser
    X_train, X_test, y_train, y_test, _ = None, None, None, None, None
    if args.prepare or args.train or args.validate:
        print("📊 Chargement des données...")
        X_train, X_test, y_train, y_test, _ = prepare_data()
        print("✅ Données préparées avec succès!")

    # Entraîner le modèle
    if args.train:
        print("🚀 Entraînement du modèle...")
        model = train_model(X_train, y_train)

        # Ensure previous MLflow run is ended
        if mlflow.active_run():
            mlflow.end_run()  # End the previous active run

        # Start a new MLflow run
        with mlflow.start_run():
            # Log the model to MLflow
            mlflow.log_param("n_estimators", 100)  # Example hyperparameter
            mlflow.log_param("max_depth", 10)  # Example hyperparameter
            mlflow.sklearn.log_model(model, "random_forest_model")

            # Save the model and log metrics to Elasticsearch
            save_model(model)
            log_to_elasticsearch(mlflow.active_run().info.run_id, {"accuracy": 0.85})  # Example metric
            print("✅ Modèle entraîné et sauvegardé!")

    # Évaluer le modèle
    if args.validate:
        print("📈 Évaluation du modèle...")
        try:
            model = load_model()  # Chargement du modèle existant
            evaluate_model(model, X_test, y_test)
            print("✅ Modèle évalué avec succès!")

            # Ensure previous MLflow run is ended
            if mlflow.active_run():
                mlflow.end_run()  # End the previous active run

            # Log evaluation results to Elasticsearch in a new MLflow run
            with mlflow.start_run():
                log_to_elasticsearch(mlflow.active_run().info.run_id, {"accuracy": 0.85})  # Example metric

        except FileNotFoundError:
            print("❌ Aucun modèle trouvé, veuillez d'abord entraîner le modèle.")


if __name__ == "__main__":
    main()
