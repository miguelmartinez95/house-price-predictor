import pandas as pd
import pytest
from src.features.engineer import create_features, create_preprocessor,run_feature_engineering

def test_create_features_missing_required_columns():
    df_invalid = pd.DataFrame({'price': [1, 2, 3]})  # missing sqft, etc.
    with pytest.raises(KeyError):
        create_features(df_invalid)

def test_create_preprocessor_structure():
    preprocessor = create_preprocessor()
    names = [name for name, _, _ in preprocessor.transformers]
    assert 'num' in names and 'cat' in names

def test_run_feature_engineering_missing_file(tmp_path):
    missing_file = tmp_path / "missing.csv"
    output_file = tmp_path / "output.csv"
    preprocessor_file = tmp_path / "preprocessor.joblib"

    with pytest.raises(FileNotFoundError):
        run_feature_engineering(missing_file, output_file, preprocessor_file)

def test_run_feature_engineering_with_empty_data(tmp_feature_paths):
    input_file, output_file, preprocessor_file = tmp_feature_paths
    pd.DataFrame().to_csv(input_file, index=False)

    with pytest.raises(ValueError):
        run_feature_engineering(input_file, output_file, preprocessor_file)
