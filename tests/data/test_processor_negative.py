# tests/processing/test_processor_negative.py
import pytest
import pandas as pd
import numpy as np
from src.data.run_processing import clean_data, load_data, process_data

def test_load_data_nonexistent_file_raises():
    with pytest.raises(FileNotFoundError):
        load_data("non_existent.csv")

def test_clean_data_missing_price_column_raises(sample_df):
    df_no_price = sample_df.drop(columns=["price"])
    with pytest.raises(KeyError):
        clean_data(df_no_price)

def test_clean_data_empty_dataframe():
    df_empty = pd.DataFrame(columns=["price", "city"])
    cleaned = clean_data(df_empty)
    assert cleaned.empty

def test_clean_data_all_nan_price_column():
    df = pd.DataFrame({"price": [np.nan, np.nan, np.nan]})
    cleaned = clean_data(df)
    # All NaNs remain since median is NaN
    assert cleaned["price"].isnull().all()

def test_process_data_invalid_input_raises(tmp_path):
    bad_input = tmp_path / "missing.csv"
    output_path = tmp_path / "output.csv"
    with pytest.raises(FileNotFoundError):
        process_data(str(bad_input), str(output_path))
