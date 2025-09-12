import numpy as np
import pandas as pd
import os
import pickle
import json
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import logging
import sys
from datetime import datetime
from typing import Tuple, Dict, Any, Optional
from sklearn.exceptions import NotFittedError


# --------------- logging setup ----------------------------
LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)

# alternate method
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# LOG_DIR = os.path.join(BASE_DIR, "logs")
# os.makedirs(LOG_DIR, exist_ok=True)

# alternate method
# BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))  # 
# LOG_DIR = os.path.join(BASE_DIR, "logs")
# os.makedirs(LOG_DIR, exist_ok=True)

# create custom handlers
stdout_handler = logging.StreamHandler(sys.stdout)  # INFO and below
stderr_handler = logging.StreamHandler(sys.stderr)  # ERROR and above

# Set log levels for handlers
stdout_handler.setLevel(logging.INFO)
stderr_handler.setLevel(logging.ERROR)

# Common log format
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
stdout_handler.setFormatter(formatter)
stderr_handler.setFormatter(formatter)

# File handler (all logs go to file)
file_handler = logging.FileHandler(os.path.join(LOG_DIR, "pipeline.log"))
file_handler.setFormatter(formatter)

# Get the root logger
logger = logging.getLogger("model_building")
logger.setLevel(logging.DEBUG)
logger.addHandler(stdout_handler)
logger.addHandler(stderr_handler)
logger.addHandler(file_handler)


# reading test data
def read_data(test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test file not found: {test_path}")
        test_data = pd.read_csv(test_path)
        if test_data.empty:
            raise ValueError(f"Test data is empty: {test_path}")
        if test_data.shape[1] < 2:
            raise ValueError(f"Test csv must contain feature + target column: {test_path}")

        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        logger.info(f"Loaded test data {test_path} with shape {test_data.shape}")
        return X_test, y_test
    
    except pd.errors.ParserError as e:
        logger.critical(f"CSV parsing error for {test_path}: {e}", exc_info=True)
        raise
    
    except Exception as e:
        logger.critical(f"Failed to read test data: {e}", exc_info=True)
        raise


def load_model(model_path: str):
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
            logger.info(f"Loaded model from {model_path}")
            return model
        
    except (pickle.UnpicklingError, EOFError) as e:
        logger.error(f"Failed to unpickle model {model_path}: {e}", exc_info=True)
        raise
        
    except Exception as e:
        logger.critical(f"Unexpected error in load_model {model_path}: {e}", exc_info=True)
        raise

def compute_scores(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Optional[float]]:
    metrics: Dict[str, Optional[float]] = {"accuracy": None, "precision": None, "recall": None, "roc_auc": None}

    try:
        # predictions
        y_pred = model.predict(X_test)

    except NotFittedError:
        logger.error("Model is not fitted.", exc_info=True)
        raise

    except Exception as e:
        logger.critical(f"Model prediction failed: {e}", exc_info=True)
        raise

    # convert to numpy arrays
    y_test_arr = np.asarray(y_test)
    y_pred_array = np.asarray(y_pred)

    # Basic metrics (handle binary vs multiclass)
    n_classes = len(np.unique(y_test_arr))

    try:
        if n_classes == 2:
            metrics['accuracy'] = float(accuracy_score(y_test_arr, y_pred_array))
            metrics['precision'] = float(precision_score(y_test_arr, y_pred_array, zero_division=0))
            metrics['recall'] = float(recall_score(y_test_arr, y_pred_array, zero_division=0))
        else:
            # multiclass: use 'macro' average
            metrics['accuracy'] = float(accuracy_score(y_test_arr, y_pred_array))
            metrics['precision'] = float(precision_score(y_test_arr, y_pred_array, average="macro", zero_division=0))
            metrics['recall'] = float(recall_score(y_test_arr, y_pred_array, average="macro", zero_division=0))

    except Exception as e:
        logger.error(f"Error computing basic metrics: {e}", exc_info=True)
        # Leave those metrics as None if computation failed

    """
    In multiclass problems, precision can be computed in different ways:
    micro → aggregate all TP/FP across classes, then compute precision (weighted by class frequency). Favors large classes.
    macro → compute precision per class, then take the unweighted mean. Treats all classes equally, regardless of imbalance.
    weighted → compute precision per class, then take a weighted mean based on class frequencies. Handles imbalance but biases toward majority class.
    "macro" is chosen here because:
    It gives equal importance to minority and majority classes.
    Useful when you want a balanced view of model performance across all classes, especially in imbalanced datasets.
    """

    # ROC AUC: try predict_proba, else decision_function; skip if unavailable
    try:
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)
            if n_classes == 2:
                # probability for positive class (assume index 1)
                score_for_roc = y_score[:, 1]
                metrics['roc_auc'] = float(roc_auc_score(y_test_arr, score_for_roc))
            else:
                metrics['roc_auc'] = float(roc_auc_score(y_test_arr, y_score, multi_class="ovr", average="macro"))

        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
            if n_classes == 2:
                metrics['roc_auc'] = float(roc_auc_score(y_test_arr, y_score))
            else:
                metrics['roc_auc'] = float(roc_auc_score(y_test_arr, y_score, multi_class="ovr", average="macro"))

        else:
            logger.warning("Model has neither predict_proba nor decision_function; skipping ROC_AUC.")
            metrics['roc_auc'] = None

    except Exception as e:
        logger.warning(f"Failed to compute ROC AUC: {e}", exc_info=True)

    return metrics

# ---- Notes ----
    """
    🔹 In multiclass problems, precision can be computed in different ways:
    micro → aggregate all TP/FP across classes, then compute precision (weighted by class frequency). Favors large classes.
    macro → compute precision per class, then take the unweighted mean. Treats all classes equally, regardless of imbalance.
    weighted → compute precision per class, then take a weighted mean based on class frequencies. Handles imbalance but biases toward majority class.
    "macro" is chosen here because:
    It gives equal importance to minority and majority classes.
    Useful when you want a balanced view of model performance across all classes, especially in imbalanced datasets.

    🔹 hasattr(model, "predict_proba")
    hasattr(obj, attr) checks if an object (model) has a method or attribute (predict_proba).
    Not all sklearn models support probability estimates (e.g., SVM without probability=True).
    If available:
    predict_proba(X) returns class probability distribution per sample.
    For binary → y_score[:, 1] = probability of the positive class.
    For multiclass → full probability matrix used with multi_class="ovr" (One-vs-Rest).
    🔹 hasattr(model, "decision_function")
    Some models (like SVMs, Logistic Regression) provide decision scores instead of probabilities.
    decision_function(X) → gives raw margin values (distance from decision boundary).
    For binary: Higher score = stronger confidence in positive class.
    For multiclass: Matrix of decision scores for each class.
    ROC-AUC works with ranked scores, so decision scores can be directly used.
    🔹 Why both?
    Some models support both (LogisticRegression).
    Some only support one:
    Tree models → predict_proba
    SVM (without probability=True) → decision_function
    Fallback ensures robustness.
    🔹 Multi-class handling
    multi_class="ovr" = One-vs-Rest strategy for ROC curves.
    "macro" averaging ensures fairness across all classes.
    """

def save_json(data: dict, filepath: str) -> None:
    """
    Save dictionary data as JSON, ensuring all NumPy types
    (scalars & arrays) are converted to JSON-serializable
    Python native types.
    """
    safe_data = {}
    for k, v in data.items():
        try:
            if isinstance(v, np.generic):
                # NumPy scalar → Python scalar
                safe_data[k] = v.item()
            elif isinstance(v, np.ndarray):
                if v.size == 1:
                    # Single element array → scalar
                    safe_data[k] = v.item()
                else:
                    # Multi-element array → list
                    safe_data[k] = v.tolist()
            else:
                safe_data[k] = v

        except Exception as e:
            logger.error(f"Failed to process key {k} with value {v}: {e}", exc_info=True)
            safe_data[k] = str(v)   # fallback to string

    try:
        with open(filepath, "w") as f:
            json.dump(safe_data, f, indent = 4)
        
        logger.info(f"Metrics saved successfully at filepath:   {filepath}")
    except Exception as e:
        logger.critical(f"Failed to save metrics json at {filepath}: {e}", exc_info=True)
        raise

    """
    JSON only supports native Python types: int, float, str, list, dict, bool, None.
    NumPy types (np.int64, np.float32, np.ndarray) are not JSON serializable.
    Using .item() ensures that scalars and 1-element arrays are safely converted into their native equivalents.
    v.item() converts NumPy scalars or 1-element arrays into native Python types (int, float,
    """

# ------------------------ Main ----------------------------
def main():

    try:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        models_dir = os.path.join(BASE_DIR, "models")
        metrics_dir = os.path.join(BASE_DIR, "reports", "metrics")
        os.makedirs(metrics_dir, exist_ok=True)

        if not os.path.exists(models_dir):
            raise FileNotFoundError(f"Models directory not found: {models_dir}")

        # Read test data once
        test_path = os.path.join(BASE_DIR, "data", "features", "test_tfidf.csv")
        X_test, y_test = read_data(test_path)

        # aggregated summary
        metrics_summary: Dict[str, Dict[str, Optional[float]]] = {}

        # iterate over latest models
        for fname in sorted(os.listdir(models_dir)):
            if not fname.endswith("_latest.pkl"):
                continue

            model_path = os.path.join(models_dir, fname)
            model_name = os.path.splitext(fname)[0]

            try:
                model = load_model(model_path)
            except Exception as e:
                logger.error(f"Skipping model {fname} due to load error: {e}", exc_info=True)
                continue

            try:
                metrics = compute_scores(model, X_test, y_test)

                # add metadata
                metrics_with_meta = {
                    "model": model_name,
                    "evaluated_at": datetime.now().isoformat(),
                    **metrics
                }

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                latest_metrics_filename = f"{model_name}_metrics_latest.json"
                ts_metrics_filename = f"{model_name}_metrics_{timestamp}.json"

                latest_metrics_path = os.path.join(metrics_dir, latest_metrics_filename)
                ts_metrics_path = os.path.join(metrics_dir, ts_metrics_filename)

                save_json(metrics_with_meta, latest_metrics_path)
                save_json(metrics_with_meta, ts_metrics_path)

                # update summary (store only numeric metrics)
                metrics_summary[model_name] = {
                    "accuracy": metrics.get("accuracy"),
                    "precision": metrics.get("precision"),
                    "recall": metrics.get("recall"),
                    "roc_auc": metrics.get("roc_auc"),
                    "evaluated_at": metrics_with_meta['evaluated_at']
                }

                logger.info(f"Evaluated {model_name}: {metrics_summary[model_name]}")

            except Exception as e:
                logger.error(f"Failed to evaluate model {model_name}: {e}", exc_info=True)
                continue
            
            # Save aggregated summary (useful to register with DVC)
            summary_path = os.path.join(metrics_dir, "metrics_summary.json")
            save_json(metrics_summary, summary_path)

            logger.info("Model_evaluation completed for all models. Mertrics written to data/metrics/")

    except Exception as e:
        logger.critical(f"Model_evaluation failed: {e}", exc_info=True)
        sys.exit(1)



if __name__ == "__main__":
    main()


    