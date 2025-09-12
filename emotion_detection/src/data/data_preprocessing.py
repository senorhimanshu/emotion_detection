import numpy as np
import pandas as pd
import os
import logging
import sys
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer

# ---------------------- logging setup -------------------------
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
logger = logging.getLogger("data_preprocessing")
logger.setLevel(logging.DEBUG)
logger.addHandler(stdout_handler)
logger.addHandler(stderr_handler)
logger.addHandler(file_handler)

try:
    nltk.download("wordnet")
    nltk.download("stopwords")
except Exception as e:
    logger.error(f"Failed to download NLTK resources: {e}", exc_info=True)
    raise

# ------------------------- Utility Functions ---------------------------

# fetch the data from data/raw
def fetch_data(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training file not found at {train_path}")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Testing file not found at {test_path}")
        
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)

        if "content" not in train_data.columns or "content" not in test_data.columns:
            raise KeyError("'content' column missing in input csv files.")
        
        logger.info(f"Train data shape: {train_data.shape}, Test data shape: {test_data.shape}")
        return train_data, test_data

    except Exception as e:
        logger.critical(f"Failed to fetch data: {e}", exc_info=True)
        raise


# -------------- text processing helpers -------------------------
def lemmatization(text: str) -> str:
    try:
        lemmatizer = WordNetLemmatizer()
        text = text.split()
        text = [lemmatizer.lemmatize(w) for w in text]
        return " ".join(text)
    
    except Exception as e:
        logger.error(f"Error in lemmatization: {e}", exc_info=True)
        return text     # return original text on failure

def remove_stopwords(text: str) -> str:
    try:
        stop_words = set(stopwords.words("english"))
        text = [i for i in str(text).split() if i not in stop_words]
        return " ".join(text)
    
    except Exception as e:
        logger.error(f"Error removing stopwords: {e}", exc_info=True)
        return text

def remove_numbers(text: str) -> str:
    text = re.sub(r"\d+", "", text)
    return text

def lower_case(text: str) -> str:
    text = text.split()
    text = [y.lower() for y in text]
    return " ".join(text)

def remove_punctuation(text: str) -> str:
    try:
        text = "".join([i for i in text if i not in string.punctuation])
        text = re.sub("\s+", " ", text)  
        return text.strip()
    
    except Exception as e:
        logger.error(f"Error removing punctuation: {e}", exc_info=True)
        return text

def remove_urls(text: str) -> str:
    url_pattern = re.compile(r"https?://\S+|www\.\S+")
    return url_pattern.sub(r"", text)

def remove_small_sentences(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df[df['content'].str.len() >= 3].reset_index(drop=True)
        return df
    except Exception as e:
        logger.error(f"Error removing small sentences: {e}", exc_info=True)
        return df

# Normalization Pipeline
def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df["content"] = df["content"].astype(str)
        df["content"] = df["content"].apply(lower_case)
        df["content"] = df["content"].apply(remove_stopwords)
        df["content"] = df["content"].apply(remove_numbers)
        df["content"] = df["content"].apply(remove_punctuation)
        df["content"] = df["content"].apply(remove_urls)
        df["content"] = df["content"].apply(lemmatization)

        logger.info(f"Data normalized successfully. Shape: {df.shape}")
        return df
    
    except Exception as e:
        logger.critical(f"Error during normalization: {e}", exc_info=True)
        raise

# def normalized_sentence(sentence):
#     sentence = lower_case(sentence)
#     sentence = remove_stopwords(sentence)
#     sentence = remove_numbers(sentence)
#     sentence = remove_punctuation(sentence)
#     sentence = remove_urls(sentence)
#     sentence = lemmatization(sentence)
#     return sentence

# ---------------------- Data Preprocessing main Pipeline ------------------------------

def main():
    try:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        train_path = os.path.join(BASE_DIR, "data", "raw", "train.csv")
        test_path = os.path.join(BASE_DIR, "data", "raw", "test.csv")

        train_data, test_data = fetch_data(train_path, test_path)

        train_processed_data = normalize_text(remove_small_sentences(train_data))
        test_processed_data = normalize_text(remove_small_sentences(test_data))

        # store the data inside data/processed
        data_path = os.path.join(BASE_DIR, "data", "processed")
        os.makedirs(data_path, exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)

        logger.info("Data Preprocessing Pipeline executed successfully !")

    except Exception as e:
        logger.critical(f"Data Preprocessing Pipeline execution failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()
