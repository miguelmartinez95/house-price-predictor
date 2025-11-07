# tests/processing/conftest.py
import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_df():
    """Sample dataset with missing values and outliers."""
    return pd.DataFrame({
        "price": [100000, 200000, 5000000, np.nan, 300000],
        "bedrooms": [3, 4, np.nan, 2, 3],
        "bathrooms": [2, 3, 2, 1, np.nan],
        "city": ["Seattle", "Bellevue", "Seattle", np.nan, "Redmond"]
    })

@pytest.fixture
def tmp_csv(tmp_path, sample_df):
    """Creates a temporary CSV input file."""
    csv_path = tmp_path / "input.csv"
    sample_df.to_csv(csv_path, index=False)
    return csv_path
