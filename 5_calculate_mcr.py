
import pandas as pd
import numpy as np
import os
import joblib
import logging
import sys
import json
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import config
import scipy.special
import matplotlib.pyplot as plt
class CustomScorer:
    def __init__(self, apply_target_transformation, target_transformation_method, fitted_lambda=None, fitted_offset=None):
        self.apply_target_transformation = apply_target_transformation
        self.target_transformation_method = target_transformation_method
        self.fitted_lambda = fitted_lambda
        self.fitted_offset = fitted_offset

    def __call__(self, estimator, X, y_true_transformed):
        y_pred_transformed = estimator.predict(X)

        if self.apply_target_transformation and self.target_transformation_method == "boxcox":
            fitted_lambda_local = self.fitted_lambda
            fitted_offset_local = self.fitted_offset

            if fitted_lambda_local is None or fitted_offset_local is None:
                logging.warning("Box-Cox parameters not provided to CustomScorer. This might indicate an issue during initialization.")
                pass # This path implies an error in initialization, not in pickling.

            if fitted_lambda_local != 0:
                safe_threshold = (-1.0 / fitted_lambda_local) + 1e-6
                if fitted_lambda_local > 0:
                    y_pred_transformed = np.maximum(y_pred_transformed, safe_threshold)
                else:
                    safe_threshold = (-1.0 / fitted_lambda_local) - 1e-6
                    y_pred_transformed = np.minimum(y_pred_transformed, safe_threshold)

            y_pred_orig = scipy.special.inv_boxcox(y_pred_transformed, fitted_lambda_local) - fitted_offset_local
            y_true_orig = scipy.special.inv_boxcox(y_true_transformed, fitted_lambda_local) - fitted_offset_local
        else:
            y_pred_orig = y_pred_transformed
            y_true_orig = y_true_transformed
        
        return -mean_squared_error(y_true_orig, y_pred_orig) # permutation_importance expects higher is better



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

def main():
    logging.info(f"--- Starting Script: 5_calculate_mcr.py ---")

    # Load data and models
    X_test = load_data(config.TEST_X_PATH, "test features")
    y_test = load_data(config.Y_TEST_PATH, "test target")
    best_estimators = load_data(config.BEST_ESTIMATORS_PATH, "best estimators")
    all_results_df = pd.read_csv(config.DETAILED_RESULTS_CSV)

    # Ensure y_test_ravel is in the correct format for calculations
    y_test_ravel = np.ravel(y_test)

    # Handle Box-Cox inverse transformation if applied
    if config.APPLY_TARGET_TRANSFORMATION and config.TARGET_TRANSFORMATION_METHOD == 'boxcox':
        fitted_lambda = getattr(config, 'BOXCOX_LAMBDA', 1.0)  # fallback
        fitted_offset = getattr(config, 'BOXCOX_OFFSET', 0.0)  # fallback
        if os.path.exists(config.BOXCOX_PARAMS_PATH):
            with open(config.BOXCOX_PARAMS_PATH, 'r') as _f:
                _params = json.load(_f)
                fitted_lambda = _params.get('lambda', fitted_lambda)
                fitted_offset = _params.get('offset', fitted_offset)
        else:
            logging.warning(f"BoxCox params file not found at {config.BOXCOX_PARAMS_PATH}. Falling back to config values.")
        
        # Apply inverse transform to y_test for true original scale
        y_test_orig = scipy.special.inv_boxcox(y_test_ravel, fitted_lambda) - fitted_offset
        # We also need to handle the filtering of infinite values that happened in 4_modeling.py
        # This part is crucial to ensure y_test_orig aligns with X_test
        finite_mask_test = np.isfinite(y_test_ravel)
        if not finite_mask_test.all():
            y_test_orig = y_test_orig[finite_mask_test]
            # X_test_fs in 4_modeling.py already handled this, so we need to re-align
            # For permutation importance, we will use the X_test that was used for prediction.
            # However, for reference model selection, we need to make sure the original y_test matches.
            # This is a bit tricky. For now, let's assume y_test_orig and X_test are aligned after 4_modeling.py.
            # A more robust solution might involve saving the filtered X_test and y_test from 4_modeling.py
            # or re-applying the filter here if needed.
            # For the purpose of MCR calculation, the X_test and y_test loaded here should correspond
            # to the data that was fed to the models for prediction.
            pass # The filtering should have already happened when X_test and y_test were loaded and processed in 4_modeling.py
    else:
        y_test_orig = y_test_ravel
    

    # Step 1: Define model class and Rashomon set
    logging.info("Step 1: Defining Rashomon Set")
    mcr_criterion = getattr(config, 'MCR_CRITERION', 'rmse')

    # Find the best performing model (reference model)
    if mcr_criterion == 'r2':
        if all_results_df.empty or 'Test R2' not in all_results_df.columns:
            logging.error("all_results_df is empty or 'Test R2' column is missing. Cannot calculate MCR.")
            sys.exit(1)
        
        best_model_row = all_results_df.loc[all_results_df['Test R2'].idxmax()]
        f_ref_r2 = best_model_row['Test R2']
        tolerance = getattr(config, 'MCR_R2_TOLERANCE', 0.01)
        
        logging.info(f"Reference Model (highest Test R2): {best_model_row['Model']} ({best_model_row['Feature Set Name']})")
        logging.info(f"Reference Model Test R2: {f_ref_r2:.4f}")
        logging.info(f"R2 Tolerance: {tolerance:.4f}")

        rashomon_set_df = all_results_df[all_results_df['Test R2'] >= f_ref_r2 - tolerance]
        logging.info(f"Found {len(rashomon_set_df)} models in the Rashomon Set (R2 >= {f_ref_r2 - tolerance:.4f}).")

    else: # Default RMSE
        if all_results_df.empty or 'Test RMSE' not in all_results_df.columns:
            logging.error("all_results_df is empty or 'Test RMSE' column is missing. Cannot calculate MCR.")
            sys.exit(1)
            
        best_model_row = all_results_df.loc[all_results_df['Test RMSE'].idxmin()]
        f_ref_loss = best_model_row['Test RMSE']
        
        # Define epsilon as 10% of the reference model's loss (RMSE)
        epsilon = 0.10 * f_ref_loss
        logging.info(f"Reference Model (lowest Test RMSE): {best_model_row['Model']} ({best_model_row['Feature Set Name']})")
        logging.info(f"Reference Model Test RMSE: {f_ref_loss:.4f}")
        logging.info(f"Epsilon (10% of reference RMSE): {epsilon:.4f}")

        rashomon_set_df = all_results_df[all_results_df['Test RMSE'] <= f_ref_loss + epsilon]
        logging.info(f"Found {len(rashomon_set_df)} models in the Rashomon Set (RMSE <= {f_ref_loss + epsilon:.4f}).")

    if rashomon_set_df.empty:
        logging.warning("Rashomon Set is empty. Cannot calculate MCR. Consider adjusting epsilon.")
        sys.exit(0)

    # Step 2: Use permutation_importance to calculate MR
    logging.info("Step 2: Calculating MR values using permutation_importance")
    
    # Instantiate custom scorer once after loading all necessary parameters
    # This ensures the scorer object is properly configured before being passed to parallelized jobs.
    custom_scorer_instance = CustomScorer(
        config.APPLY_TARGET_TRANSFORMATION,
        config.TARGET_TRANSFORMATION_METHOD,
        fitted_lambda=fitted_lambda if config.APPLY_TARGET_TRANSFORMATION and config.TARGET_TRANSFORMATION_METHOD == 'boxcox' else None,
        fitted_offset=fitted_offset if config.APPLY_TARGET_TRANSFORMATION and config.TARGET_TRANSFORMATION_METHOD == 'boxcox' else None
    )
    
    mr_values_per_feature = {}
    mcr_records = []  # To store (model, feature, mr) for raw export
    
    for index, row in rashomon_set_df.iterrows():
        model_key = f"{row['Model']}_{row['Feature Set Name']}"
        model_data = best_estimators.get(model_key) # This will now be a dict: {'model': ..., 'features': ...}
        
        if model_data is None:
            logging.warning(f"Model data for {model_key} not found in best_estimators. Skipping.")
            continue
            
        model = model_data['model']
        selected_features_for_model = model_data['features']
        
        logging.info(f"  Processing model: {model_key}")
        
        # Ensure X_test_fs aligns with the features the model was trained on
        current_X_test = X_test[selected_features_for_model]

        try:
            # `scoring` must be a callable or string. If string, it should be a scorer name (e.g., 'neg_mean_squared_error')
            # Since we need to inverse transform for MSE, we need a custom scorer.
            # The user's provided code uses 'neg_mean_squared_error' as string, but then calculates e_orig.
            # Let's stick to the user's logic by calculating e_orig and then using result.importances_mean.
            
            # Calculate e_orig for the current model on the original scale
            y_pred_transformed = model.predict(current_X_test)

            if config.APPLY_TARGET_TRANSFORMATION and config.TARGET_TRANSFORMATION_METHOD == 'boxcox':
                fitted_lambda_local = fitted_lambda
                fitted_offset_local = fitted_offset

                if fitted_lambda_local != 0:
                    safe_threshold = (-1.0 / fitted_lambda_local) + 1e-6
                    if fitted_lambda_local > 0:
                        y_pred_transformed = np.maximum(y_pred_transformed, safe_threshold)
                    else:
                        safe_threshold = (-1.0 / fitted_lambda_local) - 1e-6
                        y_pred_transformed = np.minimum(y_pred_transformed, safe_threshold)

                y_pred_orig = scipy.special.inv_boxcox(y_pred_transformed, fitted_lambda_local) - fitted_offset_local
                e_orig = mean_squared_error(y_test_orig, y_pred_orig)
            else:
                y_pred_orig = y_pred_transformed
                e_orig = mean_squared_error(y_test_orig, y_pred_orig) # Default for regression without transform
            
            # Ensure e_orig is not zero to avoid division by zero
            if e_orig == 0:
                logging.warning(f"Original loss for model {model_key} is zero. Skipping MR calculation for this model.")
                continue

            # Use the custom scorer for permutation importance to get change in MSE on original scale
            result = permutation_importance(model, current_X_test, y_test_ravel, 
                                            scoring=custom_scorer_instance, 
                                            n_repeats=config.PERMUTATION_IMPORTANCE_REPEATS, 
                                            random_state=config.RANDOM_STATE, n_jobs=-1)
            
            # To get loss increase, it's result.importances_mean[i] because:
            # CustomScorer returns -MSE, so importances_mean = (-MSE_orig) - (-MSE_switch) = MSE_switch - MSE_orig
            
            for i, feature in enumerate(current_X_test.columns):
                loss_increase = result.importances_mean[i] # (e_switch - e_orig)
                # Formula: mr = (e_orig + loss_increase) / e_orig = e_switch / e_orig
                mr = (e_orig + loss_increase) / e_orig
                
                if feature not in mr_values_per_feature:
                    mr_values_per_feature[feature] = []
                mr_values_per_feature[feature].append(mr)

                # Store for raw export
                mcr_records.append({
                    'Model': row['Model'],
                    'FeatureSet': row['Feature Set Name'],
                    'Feature': feature,
                    'MR': mr
                })

        except Exception as e:
            logging.error(f"Error calculating permutation importance for {model_key}: {e}", exc_info=True)
            continue

    if not mr_values_per_feature:
        logging.warning("No MR values calculated for any feature. Exiting.")
        sys.exit(0)

    # Step 3: Determine MCR interval
    logging.info("Step 3: Determining MCR interval")
    
    mcr_summary = {}
    for feature, mrs in mr_values_per_feature.items():
        mcr_min = min(mrs)
        mcr_max = max(mrs)
        mcr_summary[feature] = {'min': mcr_min, 'max': mcr_max}
        logging.info(f"  Feature: {feature}, MCR Range: [{mcr_min:.4f}, {mcr_max:.4f}]")

    # Identify necessary and redundant features
    logging.info("\n--- Feature Classification ---")
    necessary_features = []
    redundant_features = []

    for feature, mcr in mcr_summary.items():

        if mcr['min'] > 1.0: # Threshold of 1.1 means at least 0% increase in error if feature is permuted
            necessary_features.append(feature)
        

        if mcr['min'] < 1.00: # Threshold of 1.05 means less than 0% increase in error if feature is permuted
            redundant_features.append(feature)

    if necessary_features:
        logging.info(f"Necessary Features (mcr_min > 1.0): {', '.join(necessary_features)}")
    else:
        logging.info("No 'necessary' features identified based on current threshold.")

    if redundant_features:
        logging.info(f"Redundant Features (mcr_min < 1.0): {', '.join(redundant_features)}")
        logging.info(f"Redundant Features (mcr_min < 1.0): {', '.join(redundant_features)}")
    else:
        logging.info("No 'redundant' features identified based on current threshold.")

    # Step 4: Export to CSV
    logging.info("Step 4: Exporting MCR results to CSV")
    
    # Export Summary
    mcr_df = pd.DataFrame([
        {'Feature': f, 'MCR_Min': m['min'], 'MCR_Max': m['max']} 
        for f, m in mcr_summary.items()
    ])
    mcr_df.to_csv(config.MCR_SUMMARY_CSV, index=False)
    logging.info(f"  MCR summary saved to {config.MCR_SUMMARY_CSV}")

    # Export Raw Values
    raw_df = pd.DataFrame(mcr_records)
    raw_df.to_csv(config.MCR_RAW_VALUES_CSV, index=False)
    logging.info(f"  Raw MR values saved to {config.MCR_RAW_VALUES_CSV}")

    # --- Generate MCR Range Plot ---
    try:
        logging.info("Generating MCR Range Plot (Top 10 by MCR_Min)...")
        
        feature_mapping = {
            # 图片 A 中的特征 (MCR Rank)
            "cat__EF_scale": "EF Scale",
            "cat__archetype": "Building Archetype",
            "num__distance": "Distance",
            "cat__const_material_h_othr": "Horizontal Construction Material\n(Other)",
            "num__building_area_m2": "Building Area (m²)",
            "cat__shared_wall": "Shared Wall",
            "num__wall_fenesteration_per_back": "Wall Fenestration (Back, %)",
            "cat__retrofit_type_u": "Retrofit Type",
            "num__foundation_type_u_unc": "Uncertainty Level of Foundation Type",
            "cat__const_material_h_rglr_stone": "Horizontal Construction Material\n(Regular Stone)",
        
            # 图片 B 中额外出现的特征 (MI leaders)
            "cat__prop_val_communal": "Communal Property Value",
            "cat__foundation_type_u": "Foundation Type",
            "cat__buidling_use_before_tornado": "Building in Use or Not (Pre-tornado)",
            "num__overhang_length_u": "Overhang Length",
            "num__number_stories": "Number of Stories",
            "cat__wall_anchorage_type_u": "Wall Anchorage Type",
            "cat__prop_val_historical": "Historical Property Value",
        
            # 新增特征 (来自您提供的图片)
            "num__wall_substrate_u_unc": "Wall Substrate (Uncertainty Level)",
            "cat__owner_business": "Owner Type (Business)",
            "cat__mwfrs_u_wall": "MWFRS (Wall)",
            "cat__mwfrs_u_moment_frame": "MWFRS (Moment Frame)",
            "cat__owner_government": "Owner Type (Government)",
        
                # 对照组
            "num__random_feature": "Random Feature"
        }
        
        mcr_df['Mapped_Feature'] = mcr_df['Feature'].map(feature_mapping).fillna(mcr_df['Feature'])
        
        # Sort descending by MCR_Min, then MCR_Max, then Feature (to consistently break ties at 1.0)
        top10_df = mcr_df.sort_values(by=['MCR_Min', 'MCR_Max', 'Feature'], ascending=[False, False, True]).head(10)
        
        # Re-sort ascending internally for plotting so highest is at the top
        top10_df = top10_df.sort_values(by=['MCR_Min', 'MCR_Max', 'Feature'], ascending=[True, True, False])

        plt.rcParams.update(plt.rcParamsDefault)
        plt.rcParams.update({
            'font.family': 'Arial',
            'font.size': 18,
            'axes.titlesize': 24,
            'axes.labelsize': 20,
            'xtick.labelsize': 18,
            'ytick.labelsize': 18,
        })

        plt.figure(figsize=(14, 8))
        
        # The horizontal lines connecting min to max for each feature
        plt.hlines(y=top10_df['Mapped_Feature'], xmin=top10_df['MCR_Min'], xmax=top10_df['MCR_Max'], color='black', linewidth=2.0, zorder=1)
        
        # Scatter plots for the dots over the lines
        plt.scatter(top10_df['MCR_Min'], top10_df['Mapped_Feature'], color='#164A89', s=160, zorder=2, label=r'$MCR_{-}(\epsilon)$')
        plt.scatter(top10_df['MCR_Max'], top10_df['Mapped_Feature'], color='#F46D43', s=160, zorder=2, label=r'$MCR_{+}(\epsilon)$')
        
        # Dashed vertical line at 1.0 (Wait MR could be below 1 if permuting improves score, so plot it)
        plt.axvline(x=1.0, color='red', linestyle='--', linewidth=2.0, zorder=0, label='MR = 1.0')
        
        plt.title('Top 10 Features by Model Reliance Range', fontsize=24, pad=15, fontweight='bold')
        plt.xlabel('Model Reliance Ratio (e_perm / e_orig)', fontsize=20)
        plt.ylabel('')
        plt.legend(prop={'size': 18})
        
        plt.grid(axis='x', linestyle='--', alpha=0.3)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        # Make sure directory exists
        os.makedirs(config.BASE_RESULTS_DIR, exist_ok=True)
        mcr_plot_path = os.path.join(config.BASE_RESULTS_DIR, 'mcr_range_plot.png')
        plt.savefig(mcr_plot_path, dpi=300)
        logging.info(f"  ✅ MCR range plot saved to {mcr_plot_path}")
        plt.close()
    except Exception as e:
        logging.error(f"Failed to generate MCR range plot: {e}", exc_info=True)

    logging.info("--- Script Finished: 5_calculate_mcr.py ---")

if __name__ == '__main__':
    main()
