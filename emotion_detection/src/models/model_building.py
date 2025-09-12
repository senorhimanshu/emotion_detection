import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost

import pickle
from datetime import datetime
import os
import yaml
import logging
import sys

# -------------------- logging setup ------------------------
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

# ----------- Utility functions -----------------

def load_params(param_path: str) -> tuple[int, float, int]:
    try:
        with open(param_path, "r") as f:
            params = yaml.safe_load(f)

        models_config = params.get("model_building", {}).get("models", {})
        if not models_config:
            raise KeyError("No models specified under model_building.models in params.yaml")
        
        logger.info(f"Loaded model configuration: {list(models_config.keys())}")
        return models_config
    
    except FileNotFoundError:
        logger.error(f"Parameter file not found: {param_path}", exc_info=True)
        raise

    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {param_path}: {e}", exc_info=True)
        raise
    
    except Exception as e:
        logger.critical(f"Unexpected error in load_params: {e}", exc_info=True)
        raise


# fetch the data from data/features
def fetch_data(train_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
    
        train_data = pd.read_csv(train_path) 
        if train_data.empty:
            raise ValueError("Training data is empty")

        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        logger.info(f"Fetched training data from {train_path} with shape {train_data.shape}")
        return X_train, y_train
    
    except FileNotFoundError:
        logger.error(f"Training file not found: {train_path}", exc_info=True)
        raise

    except pd.errors.ParserError as e:
        logger.error(f"Error parsing csv file {train_path}: {e}", exc_info=True)
        raise
    
    except Exception as e:
        logger.critical(f"Unexpected error in fetch_data: {e}", exc_info=True)
        raise

def get_model(model_name, config):
    if model_name == "GradientBoosting":
        return GradientBoostingClassifier(**config)
    elif model_name == "RandomForest":
        return RandomForestClassifier(**config)
    elif model_name == "LogisticRegression":
        return LogisticRegression(**config)
    elif model_name == "XGBoost":
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(use_label_encoder = False, eval_metric = "mlogloss", **config)
        
        except ImportError:
            logger.error(f"XGBoost not installed. Please install it to use XGBoost models.")
            raise
    
    else:
        raise ValueError(f"Unsupported model: {model_name}")


# --------------- Main model_building Pipeline ------------------------

def main():

    try:

        logger.info("Starting model_building pipeline...")

        # path set up
        # Path of the current file
        # => goes 3 levels up: from src/models/ to emotion_detection/
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # relative path of params.yaml
        param_path = os.path.join(BASE_DIR, "params.yaml")
        # abs path of params.yaml
        # param_path = os.path.abspath(param_path)

        # setting up models directory
        MODELS_DIR = os.path.join(BASE_DIR, "models")
        os.makedirs(MODELS_DIR, exist_ok=True)
        train_path = os.path.join(BASE_DIR, "data", "features", "train_tfidf.csv")

        # Load configs and data
        models_config = load_params(param_path)
        X_train, y_train = fetch_data(train_path)

        # create a timestamped file name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # model_filename = f"./data/models/gb_model_{timestamp}.pkl"

        for model_name, config in models_config.items():
            try:
                logger.info(f"Training {model_name} with config: {config}")
                model = get_model(model_name, config)
                model.fit(X_train, y_train)
                logger.info(f"{model_name} training completed.")
                
                # save models (latest + timestamped)
                latest_model_path = os.path.join(MODELS_DIR, f"{model_name.lower()}_latest.pkl")
                backup_model_path = os.path.join(MODELS_DIR, f"{model_name.lower()}_{timestamp}.pkl")

                # save latest model
                with open(latest_model_path, "wb") as f:
                    pickle.dump(model, f)

                with open(backup_model_path, "wb") as f:
                    pickle.dump(model, f)

                logger.info(f"{model_name} saved at {latest_model_path} and {backup_model_path}")

            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}", exc_info=True)

        logger.info("Model_building: All model training completed !!")

    except Exception as e:
        logger.critical(f"Model Building Pipeline failed: {e}", exc_info=True)
        sys.exit(1)

# ------------------- Entry Point ---------------------------------
if __name__ == "__main__":
    main()