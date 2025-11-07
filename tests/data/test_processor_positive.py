# tests/processing/test_processor_positive.py
import pandas as pd
from src.data.run_processing import load_data, clean_data, process_data

def test_load_data_reads_valid_csv(tmp_csv):
    df = load_data(tmp_csv)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "price" in df.columns

def test_clean_data_fills_missing_values(sample_df):
    cleaned = clean_data(sample_df)
    assert cleaned.isnull().sum().sum() == 0

def test_clean_data_removes_outliers(sample_df):
    cleaned = clean_data(sample_df)
    assert cleaned["price"].max() < 1_000_000, "Outlier price should be removed"

def test_clean_data_preserves_structure(sample_df):
    cleaned = clean_data(sample_df)
    assert set(cleaned.columns) == set(sample_df.columns)
    assert all(col in cleaned for col in ["price", "bedrooms", "bathrooms", "city"])

def test_process_data_creates_output_file(tmp_csv, tmp_path):
    output_file = tmp_path / "processed" / "out.csv"
    result_df = process_data(str(tmp_csv), str(output_file))

    assert output_file.exists()
    assert isinstance(result_df, pd.DataFrame)
    assert not result_df.isnull().any().any()

def test_process_data_creates_directories(tmp_csv, tmp_path):
    nested_output = tmp_path / "deep" / "dir" / "result.csv"
    process_data(str(tmp_csv), str(nested_output))
    assert nested_output.exists()
