import pandas as pd
import joblib
from src.features.engineer import create_features, create_preprocessor, run_feature_engineering

def test_create_features_adds_expected_columns(sample_feature_df):
    df_featured = create_features(sample_feature_df)
    for col in ['house_age', 'price_per_sqft', 'bed_bath_ratio']:
        assert col in df_featured.columns

def test_house_age_is_calculated_correctly(sample_feature_df):
    current_year = pd.Timestamp.now().year
    df_featured = create_features(sample_feature_df)
    expected_ages = current_year - sample_feature_df['year_built']
    assert all(df_featured['house_age'] == expected_ages)

def test_price_per_sqft_correct(sample_feature_df):
    df_featured = create_features(sample_feature_df)
    expected = sample_feature_df['price'] / sample_feature_df['sqft']
    pd.testing.assert_series_equal(df_featured['price_per_sqft'], expected, check_names=False)

def test_bed_bath_ratio_handles_zero_bathrooms(sample_feature_df):
    df_featured = create_features(sample_feature_df)
    assert all(df_featured['bed_bath_ratio'] >= 0)
    assert df_featured.loc[sample_feature_df['bathrooms'] == 0, 'bed_bath_ratio'].eq(0).all()

def test_create_preprocessor_returns_transformer():
    preprocessor = create_preprocessor()
    assert hasattr(preprocessor, 'fit_transform')

def test_run_feature_engineering_pipeline(tmp_feature_paths, sample_feature_df):
    input_file, output_file, preprocessor_file = tmp_feature_paths
    sample_feature_df.to_csv(input_file, index=False)

    df_result = run_feature_engineering(
        input_file, output_file, preprocessor_file
    )

    # Verify files exist
    assert output_file.exists()
    assert preprocessor_file.exists()

    # Check preprocessor was saved correctly
    preprocessor = joblib.load(preprocessor_file)
    assert hasattr(preprocessor, 'transform')

    # Output should have no NaN
    assert not df_result.isnull().any().any()
