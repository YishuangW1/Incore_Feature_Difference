import pandas as pd
import numpy as np
import os
import json
import joblib
import logging
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import config

from sklearn.model_selection import train_test_split
# 导入 OrdinalEncoder
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from scipy import stats

# Set Global Random Seed
np.random.seed(config.RANDOM_STATE)


def setup_logging(log_file=config.PIPELINE_LOG_PATH):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file, mode='a'), logging.StreamHandler(sys.stdout)],
        force=True
    )


setup_logging()

# --- Dynamic Importer for Balancing Methods ---
sampler_class = None
if config.BALANCING_METHOD:
    try:
        if config.BALANCING_METHOD == 'SMOTE':
            from imblearn.over_sampling import SMOTE

            sampler_class = SMOTE
        logging.info(f"Selected balancing method: {config.BALANCING_METHOD}")
    except ImportError:
        logging.error(f"Error: 'imbalanced-learn' not found for BALANCING_METHOD '{config.BALANCING_METHOD}'.")


def filter_features(df, keywords_to_remove):
    """
    Standard feature filtering based on keywords.
    """
    cols_to_drop = {col for col in df.columns if any(keyword.lower() in col.lower() for keyword in keywords_to_remove)}
    logging.info(f"Filtering features. Removing: {sorted(list(cols_to_drop))}")
    return df.drop(columns=list(cols_to_drop), errors='ignore')


def main():
    logging.info(f"--- Starting Script: 2_dataPreprocessing.py (Refactored & Visuals Restored) ---")
    logging.info(f"--- MODE: {config.PROBLEM_TYPE.upper()} ---")

    # 1. Load Data
    df = pd.read_csv(config.CLEANED_CSV_PATH, low_memory=False)

    # ==============================================================================
    # 🛑 HANDLE INFINITE & INVALID VALUES (NEW ADDITION)
    # ==============================================================================
    logging.info("Checking for infinite values (inf / -inf) and Excel errors (e.g., #NAME?)...")

    # Check if config asks to drop inf
    invalid_strs = ['#NAME?']
    if config.DROP_INF:
        logging.info("Config DROP_INF is True. Removing rows containing infinite values or '#NAME?'...")
        original_count = len(df)
        num_cols = df.select_dtypes(include=[np.number]).columns
        is_inf_num = np.isinf(df[num_cols]).any(axis=1)
        
        # 2. Identify rows with '#NAME?' in any columns
        is_invalid_str = df.isin(invalid_strs).any(axis=1)

        # Drop rows that have either infinity or invalid strings
        df = df[~(is_inf_num | is_invalid_str)]

        dropped_count = original_count - len(df)
        if dropped_count > 0:
            logging.info(f"  -> Dropped {dropped_count} rows containing inf/-inf or invalid strings like '#NAME?'.")
        else:
            logging.info("  -> No infinite or invalid values found to drop.")
    else:
        replacement_val = config.INF_REPLACEMENT_VALUE
        logging.info(f"Config DROP_INF is False. Replacing infinite values and invalid strings with {replacement_val}...")
        df = df.replace([np.inf, -np.inf] + invalid_strs, replacement_val)

    # 2. Manual Column Fixing (Code B Logic)
    cols_to_fix = ['building_area_m2', 'roof_slope_u']
    for col in cols_to_fix:
        if col in df.columns:
            logging.info(f"Fixing '{col}': Forcing to numeric type...")
            df[col] = pd.to_numeric(df[col], errors='coerce')
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logging.info(f"  -> '{col}' fixed. Median value used for fill: {median_val}")

    # 3. Target Processing
    logging.info("  [Regression] Processing Target...")
    y = pd.to_numeric(df[config.TARGET_COLUMN], errors='coerce')
    if y.isnull().any():
        logging.warning(f"  Dropping {y.isnull().sum()} invalid NaN values in Target.")
        valid_idx = y.notnull()
        df = df[valid_idx]
        y = y[valid_idx]
    X = df.drop(columns=[config.TARGET_COLUMN])

        # [REMOVED] Box-Cox transformation moved after train_test_split to avoid data leakage.

    # 4. Filter Features (Standard Code B Logic)
    X = filter_features(X, config.KEYWORDS_TO_REMOVE_FROM_X)

    # 5. Geospatial Groups (For CV)
    groups_train = None
    if config.USE_GEOSPATIAL_CV and 'latitude' in X.columns and 'longitude' in X.columns:
        logging.info("Generating Geospatial Groups...")
        kmeans = KMeans(n_clusters=config.N_SPLITS_CV, random_state=config.RANDOM_STATE)
        X['spatial_group'] = kmeans.fit_predict(X[['latitude', 'longitude']])

    # 6. Train/Test Split
    # Since the target (e.g., incore_RPS) acts like a category (shared values), we stratify by it.
    # Convert to string to avoid float precision grouping issues during stratification.
    stratify_target = y.astype(str)
    
    # Check if any class has only 1 member, as stratify requires at least 2
    class_counts = stratify_target.value_counts()
    rare_classes = class_counts[class_counts < 2].index
    
    if len(rare_classes) > 0:
        logging.info(f"Found {len(rare_classes)} rare RPS classes with only 1 member. Dropping these samples.")
        logging.info(f"The following rare target values were dropped: {list(rare_classes)}")
        
        # Drop these rows from X and y, and stratify_target
        valid_idx = ~stratify_target.isin(rare_classes)
        X = X[valid_idx]
        y = y[valid_idx]
        stratify_target = stratify_target[valid_idx]

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE,
            stratify=stratify_target
        )
    except Exception as e:
        logging.warning(f"Stratification failed ({e}). Falling back to random split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE,
            stratify=None
        )

    if 'spatial_group' in X_train.columns:
        groups_train = X_train['spatial_group']
        X_train = X_train.drop(columns=['spatial_group'])
        X_test = X_test.drop(columns=['spatial_group'])
        joblib.dump(groups_train, os.path.join(config.DATA_DIR, 'groups_train.pkl'))

    # 7. Apply Target Transformation (AFTER Split to avoid Leakage)
    if config.APPLY_TARGET_TRANSFORMATION:
        logging.info("Applying Box-Cox Transformation to Target (Train-Test isolated)...")
        # Handle potential infinite values (though usually dropped or replaced by now)
        y_train_finite_mask = np.isfinite(y_train)
        y_train_finite = y_train[y_train_finite_mask]
        
        min_val = y_train_finite.min()
        offset = abs(min_val) + 1 if min_val <= 0 else 0
        
        # Fit Box-Cox on y_train
        y_train_transformed_vals, best_lambda = stats.boxcox(y_train_finite + offset)
        
        # Apply to y_train
        y_train_transformed = np.zeros_like(y_train, dtype=float)
        y_train_transformed[y_train_finite_mask] = y_train_transformed_vals
        if not y_train_finite_mask.all():
            y_train_transformed[~y_train_finite_mask] = y_train[~y_train_finite_mask]
        y_train = pd.Series(y_train_transformed, index=y_train.index, name=y_train.name)
        
        # Apply same parameters to y_test
        y_test_finite_mask = np.isfinite(y_test)
        y_test_finite = y_test[y_test_finite_mask]
        y_test_transformed_vals = stats.boxcox(y_test_finite + offset, lmbda=best_lambda)
        
        y_test_transformed = np.zeros_like(y_test, dtype=float)
        y_test_transformed[y_test_finite_mask] = y_test_transformed_vals
        if not y_test_finite_mask.all():
            y_test_transformed[~y_test_finite_mask] = y_test[~y_test_finite_mask]
        y_test = pd.Series(y_test_transformed, index=y_test.index, name=y_test.name)
        
        logging.info(f"  -> Box-Cox fitted on Train: Lambda={best_lambda:.6f}, Offset={offset}")
        
        # Save fitted params
        os.makedirs(os.path.dirname(config.BOXCOX_PARAMS_PATH), exist_ok=True)
        with open(config.BOXCOX_PARAMS_PATH, 'w') as f:
            json.dump({'lambda': best_lambda, 'offset': offset}, f)
        logging.info(f"  -> Box-Cox params saved to {config.BOXCOX_PARAMS_PATH}")

    # 8. Preprocessing (Standardize & Dynamic Encoding)
    logging.info(f"Preprocessing all features using {config.ENCODING_METHOD.upper()} encoding...")
    
    # 🛑 FORCE CATEGORICAL OVERRIDE: Prevent Pandas from inferring EF Scale/Event Indicator as numerical 
    for col in ['EF_scale', 'tornado_EF_unc', 'event_indicator']:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype(str)
            X_test[col] = X_test[col].astype(str)

    # for col in ['event_indicator']:
    #     if col in X_train.columns:
    #         X_train[col] = X_train[col].astype(str)
    #         X_test[col] = X_test[col].astype(str)
            
    num_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X_train.select_dtypes(include='object').columns.tolist()

    # 根据 config 开关选择编码器
    if config.ENCODING_METHOD == 'onehot':
        cat_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    elif config.ENCODING_METHOD == 'ordinal':
        # 使用 -1 处理未知类别
        cat_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    else:
        logging.error(f"Unsupported ENCODING_METHOD: {config.ENCODING_METHOD}")
        sys.exit(1)

    # 🔧 Construct Pipelines dynamically based on Imputation Strategy
    needs_imputation = getattr(config, 'MISSING_VALUE_STRATEGY', 'impute') == 'impute'

    if needs_imputation:
        logging.info("Building preprocessor WITH SimpleImputer (Train/Test isolated)...")
        # To avoid the ValueError where SimpleImputer crashes on NaNs while trying to impute infs natively:
        # We will chain three imputers: one for NaN, one for inf, one for -inf so it handles everything perfectly.
        num_pipeline = Pipeline([
            ('nan_imputer', SimpleImputer(missing_values=np.nan, strategy=getattr(config, 'NUMERICAL_IMPUTE_STRATEGY', 'mean'))),
            ('inf_imputer', SimpleImputer(missing_values=np.inf, strategy=getattr(config, 'NUMERICAL_IMPUTE_STRATEGY', 'mean'))),
            ('neg_inf_imputer', SimpleImputer(missing_values=-np.inf, strategy=getattr(config, 'NUMERICAL_IMPUTE_STRATEGY', 'mean'))),
            ('scaler', StandardScaler())
        ])
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy=getattr(config, 'CATEGORICAL_IMPUTE_STRATEGY', 'constant'), fill_value='un')),
            ('encoder', cat_encoder)
        ])
    else:
        logging.info("Building preprocessor WITHOUT imputation (NaNs were dropped)...")
        num_pipeline = Pipeline([('scaler', StandardScaler())])
        cat_pipeline = Pipeline([('encoder', cat_encoder)])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ], remainder='passthrough')

    X_train_proc_array = preprocessor.fit_transform(X_train)
    X_test_proc_array = preprocessor.transform(X_test)
    
    # Extract feature names properly from the Pipeline-wrapped ColumnTransformer
    if hasattr(preprocessor, 'get_feature_names_out'):
        feature_names = preprocessor.get_feature_names_out()
    else:
        # Fallback if sci-kit version doesn't support get_feature_names_out perfectly on this nested structure
        feature_names = [f"num__{c}" for c in num_cols] + [f"cat__{c}" for c in cat_cols]

    X_train_proc = pd.DataFrame(X_train_proc_array, columns=feature_names, index=X_train.index)
    X_test_proc = pd.DataFrame(X_test_proc_array, columns=feature_names, index=X_test.index)

    # ==============================================================================
    # 🌟 FEATURE SEPARATION LOGIC
    # ==============================================================================
    logging.info("\n=== Separating Features into Groups ===")

    def get_cols_by_keyword(df_cols, keywords):
        return [c for c in df_cols if any(k in c for k in keywords)]

    hazard_cols = get_cols_by_keyword(X_train_proc.columns, config.HAZARD_COLUMNS)
    random_cols = get_cols_by_keyword(X_train_proc.columns, ['random_feature'])
    exclude_cols = set(hazard_cols + random_cols)
    building_cols = [c for c in X_train_proc.columns if c not in exclude_cols]

    logging.info(f"  -> Hazard: {len(hazard_cols)} | Random: {len(random_cols)} | Building: {len(building_cols)}")

    print("\n" + "=" * 60)
    print(f"📢 [DEBUG] Features entering RFE (Total Count: {len(building_cols)})")
    print("-" * 60)
    print(building_cols)
    print("=" * 60 + "\n")

    # ==============================================================================
    # 🚀 RFE EXECUTION
    # ==============================================================================
    selected_building_cols = building_cols

    if config.PERFORM_RFE:
        logging.info(f"\nPerforming RFE on {len(building_cols)} Building Features...")
        X_rfe_train = X_train_proc[building_cols]

        est = RandomForestRegressor(n_estimators=150, random_state=config.RANDOM_STATE, n_jobs=-1)

        selector = RFE(est, n_features_to_select=config.N_FEATURES_TO_SELECT, step=0.0032)
        selector.fit(X_rfe_train, y_train)

        selected_building_cols = X_rfe_train.columns[selector.support_].tolist()
        logging.info(f"  -> RFE selected {len(selected_building_cols)} features.")
    else:
        logging.info("RFE is disabled.")

    # ==============================================================================
    # 📦 DATA ASSEMBLY
    # ==============================================================================
    main_cols = selected_building_cols.copy()
    if config.KEEP_HAZARD_VARIABLES:
        main_cols += hazard_cols

    main_cols += random_cols
    X_train_main = X_train_proc[main_cols]
    X_test_main = X_test_proc[main_cols]

    resid_cols = list(set(hazard_cols + selected_building_cols + random_cols))
    X_train_resid = X_train_proc[resid_cols]
    X_test_resid = X_test_proc[resid_cols]

    # ==============================================================================
    # 💾 SAVING & BALANCING
    # ==============================================================================
    y_train_orig = y_train.copy()

    if config.BALANCING_METHOD and sampler_class:
        logging.info(f"Applying {config.BALANCING_METHOD} to Main Training Data...")
        sampler = sampler_class(random_state=config.RANDOM_STATE)
        X_train_resampled, y_train_resampled = sampler.fit_resample(X_train_main, y_train)
        X_train_main = pd.DataFrame(X_train_resampled, columns=X_train_main.columns)
        y_train = y_train_resampled

    os.makedirs(config.DATA_DIR, exist_ok=True)
    joblib.dump(X_train_main, config.TRAIN_X_PATH)
    joblib.dump(y_train, config.TRAIN_Y_PATH)
    joblib.dump(X_test_main, config.TEST_X_PATH)
    joblib.dump(y_test, config.Y_TEST_PATH)
    joblib.dump(preprocessor, config.PREPROCESSOR_PATH)

    os.makedirs(config.RESIDUAL_DATA_DIR, exist_ok=True)
    joblib.dump(X_train_resid, os.path.join(config.RESIDUAL_DATA_DIR, 'X_train_full.pkl'))
    joblib.dump(X_test_resid, os.path.join(config.RESIDUAL_DATA_DIR, 'X_test_full.pkl'))
    joblib.dump(y_train_orig, os.path.join(config.RESIDUAL_DATA_DIR, 'y_train_full.pkl'))
    joblib.dump(y_test, os.path.join(config.RESIDUAL_DATA_DIR, 'y_test_full.pkl'))

    # ==============================================================================
    # 📊 Visualization
    # ==============================================================================
    logging.info("\nVisualizing data distributions (Original vs Balanced)...")
    plt.style.use(config.VISUALIZATION['plot_style'])

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    palette = config.VISUALIZATION.get('main_palette', 'viridis')

    sns.histplot(x=y, ax=axes[0, 0], kde=True)
    sns.histplot(x=y_train_orig, ax=axes[0, 1], kde=True)
    sns.histplot(x=y_train, ax=axes[1, 0], kde=True)
    sns.histplot(x=y_test, ax=axes[1, 1], kde=True)

    axes[0, 0].set_title('Target Distribution (Overall)')
    axes[0, 1].set_title('Training Data (Before Balancing)')
    axes[1, 0].set_title(f'Training Data (Final/Balanced)')
    axes[1, 1].set_title('Test Data Distribution')

    plt.tight_layout()
    plt.savefig('data_distribution_summary.png')
    logging.info("Saved data distribution summary plot: data_distribution_summary.png")

    logging.info("--- 2_dataPreprocessing.py Finished ---")


if __name__ == '__main__':
    main()
