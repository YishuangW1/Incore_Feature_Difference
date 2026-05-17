# modeling
import pandas as pd
import numpy as np
import os
import joblib
import warnings
import logging
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import config
import json
from clustering_utils import get_selected_features_by_clustering

# from dython.nominal import associations
from scipy.spatial.distance import squareform
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score
# --- Added GroupKFold import ---
from sklearn.model_selection import RepeatedKFold, KFold, GridSearchCV, StratifiedKFold, GroupKFold
from scipy.special import boxcox, inv_boxcox
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import LabelEncoder


def setup_logging(log_file=config.PIPELINE_LOG_PATH):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_file, mode='a'), logging.StreamHandler(sys.stdout)],
                        force=True)


setup_logging()


def load_data(file_path, description="data"):
    logging.info(f"Loading {description} from {file_path}...")
    try:
        return joblib.load(file_path)
    except Exception as e:
        logging.error(f"Error loading {description}: {e}", exc_info=True)
        sys.exit(1)




def plot_mutual_information(input_df):
    """
    Calculate and plot Mutual Information.
    """
    logging.info("--- Generating Mutual Information Feature Importance Plot ---")

    # 1. Prepare Data
    df = input_df.copy()
    target_col = config.TARGET_COLUMN

    if target_col not in df.columns:
        logging.warning(f"Target column {target_col} not found. Skipping MI plot.")
        return

    # 2. Separate Target and Features
    df = df.dropna(subset=[target_col])
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Remove columns based on config keywords
    keywords = config.KEYWORDS_TO_REMOVE_FROM_X
    cols_to_drop = [col for col in X.columns if any(keyword.lower() in col.lower() for keyword in keywords)]

    if cols_to_drop:
        logging.info(f"  (MI Plot) Dropping {len(cols_to_drop)} columns to match training data...")
        X = X.drop(columns=cols_to_drop, errors='ignore')

    # 3. Preprocess for Plotting (Label Encode objects)
    for col in X.select_dtypes(include=['object', 'category']).columns:
        X[col] = X[col].astype(str).fillna('missing')
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # Fill numeric NaNs
    for col in X.select_dtypes(include=np.number).columns:
        X[col] = X[col].fillna(X[col].median())

    # 4. Calculate Mutual Information
    logging.info("Calculating Mutual Information scores...")
    try:
        mi_scores = mutual_info_regression(X, y, random_state=config.RANDOM_STATE)

        mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
        mi_scores = mi_scores.sort_values(ascending=False)

        # 5. Plot
        top_n = 15
        plt.figure(figsize=(12, 8))
        top_features = mi_scores.head(top_n)
        sns.barplot(x=top_features.values, y=top_features.index, palette='viridis')

        plt.title(f'Top {top_n} Features by Mutual Information (Regression)', fontsize=16)
        plt.xlabel('Mutual Information Score')
        plt.tight_layout()

        plot_path = os.path.join(config.BASE_RESULTS_DIR, 'feature_importance_plot.png')
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Saved feature importance plot to: {plot_path}")

    except Exception as e:
        logging.error(f"Failed to generate Mutual Information plot: {e}")


def main():
    logging.info(f"--- Starting Script: 4_modeling.py ---")
    logging.info(f"--- MODE: REGRESSION ---")

    os.makedirs(config.BASE_RESULTS_DIR, exist_ok=True)

    try:
        cleaned_df = pd.read_csv(config.CLEANED_CSV_PATH, low_memory=False)
        
        # Apply the same DROP_INF logic as 2_dataPreprocessing.py to fix plotting
        invalid_strs = ['#NAME?']
        if getattr(config, 'KEEP_INF', False):
            cleaned_df = cleaned_df.replace(invalid_strs, -np.inf)
        else:
            if getattr(config, 'DROP_INF', False):
                num_cols = cleaned_df.select_dtypes(include=[np.number]).columns
                is_inf_num = np.isinf(cleaned_df[num_cols]).any(axis=1)
                is_invalid_str = cleaned_df.isin(invalid_strs).any(axis=1)
                cleaned_df = cleaned_df[~(is_inf_num | is_invalid_str)]
            else:
                replacement_val = getattr(config, 'INF_REPLACEMENT_VALUE', -999)
                cleaned_df = cleaned_df.replace([np.inf, -np.inf] + invalid_strs, replacement_val)
            
        plot_mutual_information(cleaned_df)
    except Exception as e:
        logging.warning(f"Could not load cleaned data for plotting MI: {e}")

    X_train = load_data(config.TRAIN_X_PATH, "training features")
    y_train = load_data(config.TRAIN_Y_PATH, "training target")
    X_test = load_data(config.TEST_X_PATH, "test features")
    y_test = load_data(config.Y_TEST_PATH, "test target")

    # ==============================================================================
    # 🔍 [DEBUG CHECK] Output Modeling Variable Info (Please copy this code)
    # ==============================================================================
    logging.info("\n" + "!" * 60)
    logging.info("🔍 [DATA INTEGRITY CHECK] Verifying Input Features for Modeling")
    logging.info("!" * 60)

    # 1. Output Feature Count (Most intuitive metric)
    logging.info(f"📊 Feature Count: {X_train.shape[1]}")

    # 2. Output Feature Names (Sorted for easy visual comparison)
    #    If variable names differ by even a single character, it will show here.
    sorted_cols = sorted(X_train.columns.tolist())
    logging.info(f"📝 Feature Names (Sorted):\n{sorted_cols}")

    # 3. (Advanced) Output Data Content "Fingerprint" (Checksum/Hash)
    #    Core function: Even if variable names are identical, if one uses mean imputation
    #    and another uses median, this hash will be completely different.
    #    If the Hash of two versions is the same, the data is 100% identical.
    try:
        data_hash = pd.util.hash_pandas_object(X_train).sum()
        logging.info(f"🔐 Data Content Hash (Fingerprint): {data_hash}")
    except Exception as e:
        logging.warning(f"Could not calculate hash: {e}")

    logging.info("!" * 60 + "\n")
    # ==============================================================================

    # ... (Original Logic: y_train_ravel = np.ravel(y_train) ...)

    # === [NEW] Load Geospatial Groups ===
    groups_train = None
    groups_path = os.path.join(config.DATA_DIR, 'groups_train.pkl')
    # Only load if Geospatial CV is enabled and file exists
    if config.USE_GEOSPATIAL_CV:
        if os.path.exists(groups_path):
            groups_train = joblib.load(groups_path)
            logging.info("Successfully loaded spatial groups for GroupKFold.")
        else:
            logging.warning("USE_GEOSPATIAL_CV is True but groups_train.pkl not found. Please check preprocessing.")
    # ====================================

    y_train_ravel = np.ravel(y_train)
    y_test_ravel = np.ravel(y_test)

    # === [NEW] Filter out infinite target values safely before passing to sklearn ===
    finite_mask_train = np.isfinite(y_train_ravel)
    if not finite_mask_train.all():
        logging.warning(f"  Filtering out {np.sum(~finite_mask_train)} infinite target values from Training data.")
        y_train_ravel = y_train_ravel[finite_mask_train]
        X_train = X_train[finite_mask_train]
        if groups_train is not None:
            groups_train = groups_train[finite_mask_train]

    finite_mask_test = np.isfinite(y_test_ravel)
    if not finite_mask_test.all():
        logging.warning(f"  Filtering out {np.sum(~finite_mask_test)} infinite target values from Test data.")
        y_test_ravel = y_test_ravel[finite_mask_test]
        X_test = X_test[finite_mask_test]
    # ==============================================================================

    all_results = []
    best_estimators = {}

    for threshold in config.CLUSTERING_THRESHOLDS_TO_TEST:
        feature_set_label = f"Clustered (Thresh={threshold})" if threshold is not None else "Original Features"
        logging.info(f"\n===== PROCESSING FEATURE SET: {feature_set_label} =====")

        if config.CLUSTERING_LINKAGE_METHOD == 'spearman':
            from clustering_utils import get_selected_features_spearman
            selected_features = get_selected_features_spearman(X_train, threshold)
        elif config.CLUSTERING_LINKAGE_METHOD == 'spearman_ratio':
            from clustering_utils import get_selected_features_spearman_corr
            selected_features = get_selected_features_spearman_corr(X_train, threshold)
        else:
            selected_features = get_selected_features_by_clustering(X_train, threshold, config.CLUSTERING_LINKAGE_METHOD)
            
        X_train_fs = X_train[selected_features]
        X_test_fs = X_test.reindex(columns=X_train_fs.columns, fill_value=0)

        for model_name, model_template in config.MODELS_TO_BENCHMARK.items():
            logging.info(f"  --- Benchmarking Model: {model_name} ---")
            param_grid = config.PARAM_GRIDS.get(model_name, {})

            # =====================================================
            # 1. Set Cross-Validation (CV) Strategy
            # =====================================================
            cv_method = None

            if config.USE_GEOSPATIAL_CV:
                # --- Case A: Geospatial CV enabled (Highest Priority) ---
                if groups_train is None:
                    logging.error("Config enabled Geospatial CV, but groups_train is None!")
                    sys.exit(1)

                # Dynamic n_splits adjustment
                n_groups = len(np.unique(groups_train))
                actual_n_splits = min(config.N_SPLITS_CV, n_groups)

                if actual_n_splits < 2:
                    logging.error(f"Cannot perform CV with only {n_groups} group(s). Minimum 2 groups required.")
                    continue

                if actual_n_splits < config.N_SPLITS_CV:
                    if model_name == list(config.MODELS_TO_BENCHMARK.keys())[0]:
                        logging.warning(
                            f"  [Geospatial CV] Reduced n_splits from {config.N_SPLITS_CV} to {actual_n_splits} due to limited groups ({n_groups} groups found).")

                cv_method = GroupKFold(n_splits=actual_n_splits)

                # Only print hint when running the first model
                if model_name == list(config.MODELS_TO_BENCHMARK.keys())[0]:
                    logging.info(f"  -> [Geospatial Mode] Using GroupKFold with {actual_n_splits} splits.")

            else:
                # --- Case B: Geospatial CV disabled (Fallback to old logic) ---
                cv_method = RepeatedKFold(n_splits=config.N_SPLITS_CV, n_repeats=config.N_REPEATS,
                                          random_state=config.RANDOM_STATE)

                if model_name == list(config.MODELS_TO_BENCHMARK.keys())[0]:
                    logging.info(f"  -> [Standard Mode] Using Random CV ({type(cv_method).__name__}).")

            # =====================================================
            # 2. Initialize GridSearchCV
            # =====================================================
            grid_search = GridSearchCV(estimator=model_template, param_grid=param_grid,
                                       scoring=config.GRIDSEARCH_SCORING_METRIC,
                                       cv=cv_method, n_jobs=-1, error_score='raise')

            try:
                # =====================================================
                # 3. Train Model (Fit)
                # =====================================================
                if config.USE_GEOSPATIAL_CV:
                    grid_search.fit(X_train_fs, y_train_ravel, groups=groups_train)
                else:
                    grid_search.fit(X_train_fs, y_train_ravel)

                # =====================================================
                # 4. Get Results and Calculate Metrics
                # =====================================================
                best_estimator = grid_search.best_estimator_
                combo_key = f"{model_name}_{feature_set_label}"
                # Store both the model and the features it was trained on
                best_estimators[combo_key] = {'model': best_estimator, 'features': selected_features}

                result_row = {
                    "Model": model_name, "Feature Set Name": feature_set_label,
                    "Number of Features": len(selected_features), "Threshold Value": threshold,
                    "Best Params": str(grid_search.best_params_)
                }

                y_pred_test = best_estimator.predict(X_test_fs)
                y_pred_train = best_estimator.predict(X_train_fs)

                # --- Metric Calculation Branch ---

                if config.APPLY_TARGET_TRANSFORMATION and config.TARGET_TRANSFORMATION_METHOD == 'boxcox':
                    # === Box-Cox Inverse Transformation Logic ===
                    # Load the actual fitted lambda and offset that 2_dataPreprocessing.py used.
                    # This avoids the mismatch between the auto-fitted lambda and config.BOXCOX_LAMBDA.
                    fitted_lambda = getattr(config, 'BOXCOX_LAMBDA', 1.0)  # fallback
                    fitted_offset = getattr(config, 'BOXCOX_OFFSET', 0.0)  # fallback
                    if os.path.exists(config.BOXCOX_PARAMS_PATH):
                        with open(config.BOXCOX_PARAMS_PATH, 'r') as _f:
                            _params = json.load(_f)
                            fitted_lambda = _params.get('lambda', fitted_lambda)
                            fitted_offset = _params.get('offset', fitted_offset)
                    else:
                        logging.warning(f"BoxCox params file not found at {config.BOXCOX_PARAMS_PATH}. Falling back to config values.")

                    if fitted_lambda != 0:
                        safe_threshold = (-1.0 / fitted_lambda) + 1e-6
                        if fitted_lambda > 0:
                            y_pred_test = np.maximum(y_pred_test, safe_threshold)
                            y_pred_train = np.maximum(y_pred_train, safe_threshold)
                            y_train_ravel = np.maximum(y_train_ravel, safe_threshold)
                            y_test_ravel = np.maximum(y_test_ravel, safe_threshold)
                        else:
                            safe_threshold = (-1.0 / fitted_lambda) - 1e-6
                            y_pred_test = np.minimum(y_pred_test, safe_threshold)
                            y_pred_train = np.minimum(y_pred_train, safe_threshold)
                            y_train_ravel = np.minimum(y_train_ravel, safe_threshold)
                            y_test_ravel = np.minimum(y_test_ravel, safe_threshold)

                    # Inverse Transform using ACTUAL fitted lambda/offset
                    y_pred_test_orig = inv_boxcox(y_pred_test, fitted_lambda) - fitted_offset
                    y_test_orig = inv_boxcox(y_test_ravel, fitted_lambda) - fitted_offset
                    y_pred_train_orig = inv_boxcox(y_pred_train, fitted_lambda) - fitted_offset
                    y_train_orig = inv_boxcox(y_train_ravel, fitted_lambda) - fitted_offset

                else:
                    # No transformation applied
                    y_pred_test_orig = y_pred_test
                    y_test_orig = y_test_ravel
                    y_pred_train_orig = y_pred_train
                    y_train_orig = y_train_ravel

                # Calculate Metrics
                result_row['Train R2'] = r2_score(y_train_orig, y_pred_train_orig)
                result_row['Test R2'] = r2_score(y_test_orig, y_pred_test_orig)
                result_row['Test RMSE'] = np.sqrt(mean_squared_error(y_test_orig, y_pred_test_orig))

                all_results.append(result_row)

            except Exception as e:
                logging.error(f"    ERROR running {model_name}: {e}", exc_info=True)
                continue

    all_results_df = pd.DataFrame(all_results)
    all_results_df.to_csv(config.DETAILED_RESULTS_CSV, index=False)
    joblib.dump(best_estimators, config.BEST_ESTIMATORS_PATH)

    logging.info(f"\nResults saved. Best Estimators saved.")
    print(all_results_df.to_string())

    # --- Plotting ---
    if not all_results_df.empty:
        plt.figure(figsize=(16, 8))
        metric_to_plot = 'Test R2'
        
        # Sort data for consistent plotting
        plot_df = all_results_df.sort_values(metric_to_plot, ascending=False).copy()
        
        ax = sns.barplot(x=metric_to_plot, y='Model', data=plot_df, palette='viridis')
        
        # Add numerical labels to the bars as percentages with 4 decimal places
        for i, p in enumerate(ax.patches):
            width = p.get_width()
            ax.text(width + 0.01, p.get_y() + p.get_height()/2, 
                    f'{width*100:.4f}%', 
                    va='center', fontsize=14, fontweight='bold', color='black') # Increased from 11 to 14

        plt.title(f'Model Comparison by {metric_to_plot} - Test Data | Target: {config.TARGET_COLUMN}', fontsize=26) # Increased from 18
        plt.xlabel(metric_to_plot, fontsize=22) # Increased from 14
        plt.ylabel('Model', fontsize=22) # Increased from 14
        
        # Increase tick label sizes
        ax.tick_params(axis='both', which='major', labelsize=18)
        
        # Adjust x-limit to fit the larger labels
        curr_xlim = ax.get_xlim()
        ax.set_xlim(curr_xlim[0], curr_xlim[1] * 1.6)
        
        plt.tight_layout()
        plt.savefig(os.path.join(config.BASE_RESULTS_DIR, "model_comparison_main_metric.png"))
        plt.close()
        logging.info(f"Saved model comparison plot: model_comparison_main_metric.png")

    logging.info("\n--- Script Finished ---")


if __name__ == '__main__':
    main()