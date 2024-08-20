import pandas as pd 
import logging
import numpy as np

def convert_to_float(value):
    if pd.isna(value) or (isinstance(value, str) and value.strip() == ''):
        return np.nan
    if isinstance(value, str):
        try:
            return float(value.replace(',', '.'))
        except ValueError as e:
            logging.error(f"Error converting value {value} to float: {e}")
            return np.nan
    elif isinstance(value, (int, float)):
        return float(value)
    else:
        logging.error(f"Unexpected data type for conversion: {type(value)}")
        return np.nan