import pytest
import pandas as pd
import numpy as np
from pathlib import Path

@pytest.fixture
def sample_feature_df():
    """Fixture: sample dataset for feature engineering tests."""
    return pd.DataFrame({
        'price': [300000, 450000, 600000],
        'sqft': [1500, 2000, 2500],
        'bedrooms': [3, 4, 5],
        'bathrooms': [2, 3, 0],  # note: one zero for division test
        'year_built': [2000, 1990, 2010],
        'location': ['Seattle', 'Redmond', 'Bellevue'],
        'condition': ['Good', 'Excellent', 'Fair']
    })

@pytest.fixture
def tmp_feature_paths(tmp_path):
    """Fixture: create temporary file paths for pipeline output."""
    input_file = tmp_path / "cleaned.csv"
    output_file = tmp_path / "engineered.csv"
    preprocessor_file = tmp_path / "preprocessor.joblib"
    return input_file, output_file, preprocessor_file
