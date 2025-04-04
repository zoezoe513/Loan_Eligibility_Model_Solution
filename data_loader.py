import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

def load_data(path='credit.csv'):
    try:
        df = pd.read_csv(path)
        logging.info("Dataset loaded successfully.")
        return df
    except FileNotFoundError:
        logging.error("Dataset file not found.")
        raise
