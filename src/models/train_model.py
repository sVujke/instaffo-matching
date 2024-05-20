import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from joblib import dump
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

from src.util.paths import Paths
from src.util.logger import logger


def create_features_and_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Preprocess the data by splitting into features and target.

    :param df: DataFrame containing the dataset.
    :return: Tuple containing features (X) and target (y).
    """
    X = df.drop(columns=["label"])
    y = df["label"]
    return X, y


def build_pipeline() -> Pipeline:
    """
    Build a machine learning pipeline with preprocessing and model.

    :return: Configured machine learning pipeline.
    """
    scaler = StandardScaler()
    classifier = LogisticRegression()
    pipeline = Pipeline([("scaler", scaler), ("classifier", classifier)])
    return pipeline


def log_confusion_matrix(y_true: pd.Series, y_pred: pd.Series) -> None:
    """
    Plot and log the confusion matrix.

    :param y_true: True labels.
    :param y_pred: Predicted labels.
    """
    cm = confusion_matrix(y_true, y_pred)

    logger.info("Confusion Matrix:\n" + str(cm))


def evaluate_model(
    pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series
) -> None:
    """
    Evaluate the model using cross-validation and log the scores.

    :param pipeline: Machine learning pipeline.
    :param X_train: Training features.
    :param y_train: Training labels.
    """
    p_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="precision")
    r_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="recall")
    logger.info(f"Cross-Validation Precision Scores: {p_scores}")
    logger.info(f"Cross-Validation Recall Scores: {r_scores}")
    logger.info(f"Average Precision: {p_scores.mean()}")
    logger.info(f"Average Recall: {r_scores.mean()}")


def train_model(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> None:
    """
    Train the model on the training data.

    :param pipeline: Machine learning pipeline.
    :param X_train: Training features.
    :param y_train: Training labels.
    """
    pipeline.fit(X_train, y_train)
    logger.info("Model training completed")


def save_model(pipeline: Pipeline, model_path: str) -> None:
    """
    Save the trained model to disk.

    :param pipeline: Trained machine learning pipeline.
    :param model_path: Path to save the model.
    """
    dump(pipeline, model_path)
    logger.info(f"Model saved to {model_path}")


def evaluate_on_test_set(
    pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> None:
    """
    Evaluate the model on the test set and log the performance metrics.

    :param pipeline: Machine learning pipeline.
    :param X_test: Test features.
    :param y_test: Test labels.
    """
    y_pred = pipeline.predict(X_test)
    y_score = pipeline.predict_proba(X_test)[:, 1]  # score for ranking purposes
    logger.info(f"probabilities sample {y_score[:20]}")
    logger.info("\n" + classification_report(y_test, y_pred))
    logger.info(f"ROC-AUC Score: {roc_auc_score(y_test, y_score)}")


if __name__ == "__main__":
    df = pd.read_csv(Paths.processed_dataset_path)
    logger.info(f"Data loaded from {Paths.processed_dataset_path}")

    logger.info(f"Data Info:\n{df.info()}")
    logger.info(f"First few rows:\n{df.head().T}")

    # Create features and labels
    X, y = create_features_and_labels(df)

    # Split data for evaluation purposes
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Build the pipeline
    pipeline = build_pipeline()

    # Train the model on the training data
    train_model(pipeline, X_train, y_train)

    # Evaluate on the test set
    evaluate_on_test_set(pipeline, X_test, y_test)

    # Train on the entire dataset and save the model
    train_model(pipeline, X, y)
    save_model(pipeline, Paths.match_model_path)
