import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer 
import os
import yaml
import logging
import sys

# -------------------- logging setup ----------------------------
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
logger = logging.getLogger("feature_engineering")
logger.setLevel(logging.DEBUG)
logger.addHandler(stdout_handler)
logger.addHandler(stderr_handler)
logger.addHandler(file_handler)


# ---------------------------- Utility Functions ---------------------------------

def load_params(param_path: str) -> int:
    try:
        with open(param_path, "r") as f:
            params = yaml.safe_load(f)

        max_features = params.get("feature_engineering", {}).get("max_features", None)
        if max_features is None:
            raise KeyError("Missing 'feature_engineering.max_features' in params.yaml")
        logger.info(f"Loaded max_features size = {max_features} from {param_path}")
        return max_features
    
    except FileNotFoundError:
        logger.error(f"Parameter file not found: {param_path}", exc_info=True)
        raise

    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {param_path}: {e}", exc_info=True)
        raise
    
    except Exception as e:
        logger.critical(f"Unexpected error in load_params: {e}", exc_info=True)
        raise


# fetch the data from data/processed
def fetch_data(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Train file not found at {train_path}")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test file not found at {test_path}")
        
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)

        if "content" not in train_data.columns or "sentiment" not in train_data.columns:
            raise KeyError("'content' or 'sentiment' column missing in training data.")
        
        if "content" not in test_data.columns or "sentiment" not in test_data.columns:
            raise KeyError("'content' or 'sentiment' column missing in test data.")
        
        train_data.fillna("", inplace=True)
        test_data.fillna("", inplace = True)

        logger.info(f"Train data shape: {train_data.shape}, Test data shape: {test_data.shape}")
        return train_data, test_data
    
    except pd.errors.ParserError as e:
        logger.critical(f"Error parsing csv files: {e}", exc_info=True)
        raise
    
    except Exception as e:
        logger.critical(f"Unexpected error in fetch_data: {e}", exc_info=True)
        raise

# train_data = pd.read_csv("./data/processed/train_processed.csv")
# test_data = pd.read_csv("./data/processed/test_processed.csv")

# ----------------------- Main Feature Engineering Pipeline ---------------------------------
def main():
    try:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        train_path = os.path.join(BASE_DIR, "data", "processed", "train_processed.csv")
        test_path = os.path.join(BASE_DIR, "data", "processed", "test_processed.csv")

        param_path = os.path.join(BASE_DIR, "params.yaml")
        # param_path = os.path.abspath(param_path)

        train_data, test_data = fetch_data(train_path, test_path)
        
        X_train = train_data["content"].values
        y_train = train_data['sentiment'].values

        X_test = test_data["content"].values
        y_test = test_data['sentiment'].values

        # apply bag of words
        vectorizer = CountVectorizer()
        max_features = load_params(param_path)
        vectorizer = CountVectorizer(max_features = max_features) # limit the number of features to 5000

        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)

        train_df = pd.DataFrame(X_train_bow.toarray())
        test_df = pd.DataFrame(X_test_bow.toarray())  

        train_df['label'] = y_train
        test_df['label'] = y_test

        # store the data inside data/features
        data_path = os.path.join(BASE_DIR, "data", "features")
        os.makedirs(data_path, exist_ok=True)

        train_df.to_csv(os.path.join(data_path, "train_bow.csv"), index=False)
        test_df.to_csv(os.path.join(data_path, "test_bow.csv"), index=False)

        logger.info('Feature engineering completed successfully !')

    except Exception as e:
        logger.critical(f"Feature engineering pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

