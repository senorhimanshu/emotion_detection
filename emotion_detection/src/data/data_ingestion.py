import numpy as np
import pandas as pd
import os 
import yaml
from sklearn.model_selection import train_test_split
import sys
import logging

# ---------------------- logging setup ---------------------------
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
logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.DEBUG)
logger.addHandler(stdout_handler)
logger.addHandler(stderr_handler)
logger.addHandler(file_handler)

# ---------------- Utility Functions ------------------

def load_params(param_path: str) -> float:

    try:
        with open(param_path, "r") as f:
            params = yaml.safe_load(f)
        
        test_size = params.get("data_ingestion", {}).get("test_size", 0.2)
        logger.info(f"Loaded test_size: {test_size} from {param_path}")
        return test_size

    except FileNotFoundError:
        logger.error(f"Parameter file not found: {param_path}", exc_info=True)
        raise 
    
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {param_path}: {e}", exc_info=True)
        raise 
    
    except Exception as e:
        logger.error(f"Unexpected error in load_params: {e}", exc_info=True)
        raise 


def read_data(url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url)
        if df.empty:
            raise ValueError("Loaded dataset is empty.")
        logger.info(f"Successfully loaded dataset with shape {df.shape} from {url}")
        return df

    except FileNotFoundError:
        logger.error(f"Data file not found at {url}", exc_info=True)
        raise 
    
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing csv at {url}: {e}", exc_info=True)
        raise 
    
    except Exception as e:
        logger.error(f"Unexpected error in read_data: {e}", exc_info=True)
        raise 
    

def process_data(df: pd.DataFrame) -> pd.DataFrame:

    try:
        if "tweet_id" in df.columns:
            df.drop(columns = ["tweet_id"], inplace = True)
            logger.info("Dropped 'tweet_id' column from dataset.")
        else:
            raise KeyError("Column 'tweet_id' not found in dataset.")
        
        if "sentiment" not in df.columns:
            raise KeyError("Column 'sentiment' not found in dataset.")
        
        final_df = df[df['sentiment'].isin(["happiness", "sadness"])].copy()
        final_df["sentiment"].replace({"happiness": 1, "sadness": 0}, inplace = True)

        if final_df.empty:
            raise ValueError('Processed dataframe is empty after filtering sentiments.')
        
        logger.info(f"Processed dataset shape after filtering: {final_df.shape}")
        return final_df

    except Exception as e:
        logger.error(f"Unexpected error in process_data: {e}", exc_info=True)
        raise 


def save_data(data_path: str, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:

    try:
        os.makedirs(data_path, exist_ok = True)
        train_data.to_csv(os.path.join(data_path, "train.csv"), index = False)
        test_data.to_csv(os.path.join(data_path, "test.csv"), index = False)
        logger.info(f"Saved train and test data to {data_path}")

    except Exception as e:
        logger.error(f"Unexpected error in save_data: {e}", exc_info=True)
        raise

# --------------------- Data Ingestion main pipeline -----------------------
def main():

    try:
        logger.info("Starting data ingestion process...")

        BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        param_path = os.path.join(BASE_DIR, "params.yaml")
        # param_path = os.path.abspath(param_path)

        test_size = load_params(param_path)
        df = read_data("https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv")
        final_df = process_data(df)

        train_data, test_data = train_test_split(final_df, test_size = test_size, random_state = 42)
        logger.info(f"Split data into train ({train_data.shape}) and test ({test_data.shape}) sets.")

        data_path = os.path.join(BASE_DIR, "data", "raw")
        save_data(data_path, train_data, test_data)

        logger.info("Data ingestion completed successfully.")

    except Exception as e:
        logger.critical(f"Pipeline failed: {e}", exc_info=True)     # exc_info = exception information (traceback) / exc_info=True → Logs traceback + error message / With exc_info=True, you also see where exactly in the code it failed (file, line, function).
        raise



if __name__ == "__main__":
    main()