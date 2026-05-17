import os

# ==========================================
# CRITICAL: Prevent OpenBLAS / OMP Memory Crashes
# Must be set before importing numpy/pandas/sklearn
# ==========================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
import joblib
import logging
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
import config
from clustering_utils import get_selected_features_by_clustering


# --- Logging Setup ---
def setup_logging(log_file=config.PIPELINE_LOG_PATH):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file, mode='a'), logging.StreamHandler(sys.stdout)],
        force=True
    )


setup_logging()


def load_data(file_path, description="data"):
    try:
        return joblib.load(file_path)
    except Exception as e:
        logging.error(f"Error loading {description}: {e}")
        sys.exit(1)


def run_mutual_info_analysis(X_train, y_train, label="original"):
    logging.info(f"--- Running Mutual Information Analysis ({label}) ---")

    # Convert to DataFrame for easier handling
    X = X_train.copy()
    y = np.ravel(y_train)

    # 1. We already have 'random_feature' created in 1_dataCleaning.py
    # No need to inject new random features here.

    # 2. Identify discrete feature mask
    # Preprocessed data is usually numerical; determine if a feature is discrete by its unique value count
    discrete_mask = []
    for col in X.columns:
        # If unique values are fewer than 20, treat it as a discrete categorical feature
        if X[col].nunique() < 20:
            discrete_mask.append(True)
        else:
            discrete_mask.append(False)

    # Filter out infinite target values safely before passing to sklearn
    finite_mask = np.isfinite(y)
    if not finite_mask.all():
        logging.warning(f"  Filtering out {np.sum(~finite_mask)} infinite target values before MI calculation.")
        X = X[finite_mask]
        y = y[finite_mask]

    # 3. Calculate MI Score (Averaged over iterations for stability)
    n_repeats = 30
    logging.info(f"Calculating Mutual Information scores (averaged over {n_repeats} repeats)...")

    all_mi_scores = []

    for i in range(n_repeats):
        # Shift the seed for each repeat to ensure different noise injection
        seed = config.RANDOM_STATE + i if (
                    hasattr(config, 'RANDOM_STATE') and config.RANDOM_STATE is not None) else 42 + i

        scores = mutual_info_regression(X, y, discrete_features=discrete_mask, random_state=seed)

        all_mi_scores.append(scores)

    # Average the scores across all repeats
    mi_scores = np.mean(all_mi_scores, axis=0)

    mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

    # Save to CSV
    csv_filename = f'feature_importance_mi_scores_{label}.csv'
    csv_output_path = os.path.join(config.BASE_RESULTS_DIR, csv_filename)
    os.makedirs(config.BASE_RESULTS_DIR, exist_ok=True)
    mi_series.to_csv(csv_output_path, header=['Mutual Information Score'])
    logging.info(f"MI Scores saved to {csv_output_path}")

    # 4. Determine random baseline
    # After preprocessing, the random feature column is named 'num__random_feature'
    random_col = next((c for c in mi_series.index if 'random_feature' in c), None)
    if random_col is not None:
        random_baseline = mi_series[random_col]
        logging.info(f"Random Baseline MI Score ({random_col}): {random_baseline:.4f}")
    else:
        random_baseline = 0
        logging.info("Random feature not found. Baseline set to 0.")

    # 5. Visualization
    top_n = getattr(config, 'MI_TOP_N', 10)
    top_plot_data = mi_series.head(top_n)

    # Adjust plot height based on number of features
    plot_height = max(6, len(top_plot_data) * 0.5)
    plt.figure(figsize=(10, plot_height))

    colors = ['#ff7f0e' if x <= random_baseline else '#1f77b4' for x in top_plot_data.values]

    feature_mapping = {
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
        "cat__prop_val_communal": "Communal Property Value",
        "cat__foundation_type_u": "Foundation Type",
        "cat__buidling_use_before_tornado": "Building in Use or Not (Pre-tornado)",
        "num__overhang_length_u": "Overhang Length",
        "num__number_stories": "Number of Stories",
        "cat__wall_anchorage_type_u": "Wall Anchorage Type",
        "cat__prop_val_historical": "Historical Property Value",
        "num__random_feature": "Random Feature",
        "cat__wall_cladding_u": "Wall Cladding",
        "num__first_floor_elevation_m": "First Floor Elevation (m)",
        "cat__const_material_h_brick": "Horizontal Construction Material\n(Brick)",
        "cat__prop_val_evidential": "Evidential Property Value",
        "num__roof_system_u_unc": "Uncertainty Level of Roof System",
        "num__wall_substrate_u_unc": "Wall Substrate (Uncertainty Level)",
        "cat__owner_business": "Owner Type (Business)",
        "cat__mwfrs_u_wall": "MWFRS (Wall)",
        "cat__mwfrs_u_moment_frame": "MWFRS (Moment Frame)",
        "cat__owner_government": "Owner Type (Government)",
        "num__latitude": "Latitude",
        "num__longitude": "Longitude"
    }

    mapped_index = [feature_mapping.get(idx, idx) for idx in top_plot_data.index]

    ax = sns.barplot(x=top_plot_data.values, y=mapped_index, palette=colors)
    plt.axvline(x=random_baseline, color='red', linestyle='--', label=f'Random Baseline ({random_baseline:.4f})')

    plt.title(f'Top {len(top_plot_data)} Feature Importance via Mutual Information',
              fontsize=20)  # Increased from 14
    plt.xlabel('Mutual Information Score', fontsize=16)  # Added fontsize
    plt.ylabel('', fontsize=16)  # Remove the 'None' label

    # Increase tick label sizes
    plt.tick_params(axis='both', which='major', labelsize=12)

    # Adjust x-limit to prevent legend/text overlap
    curr_xlim = plt.gca().get_xlim()
    plt.gca().set_xlim(curr_xlim[0], curr_xlim[1] * 1.1)

    plt.legend(fontsize=12)  # Increased legend font
    plt.tight_layout()

    # Save results
    output_filename = f'feature_importance_mi_top_{top_n}_{label}.png'
    output_path = os.path.join(config.BASE_RESULTS_DIR, output_filename)
    os.makedirs(config.BASE_RESULTS_DIR, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    logging.info(f"MI Plot saved to {output_path}")
    plt.close()

    # Output features exceeding the random baseline
    significant_features = mi_series[mi_series > random_baseline]
    logging.info(f"Found {len(significant_features)} features above random baseline.")
    print("\nTop Significant Features:")
    print(significant_features.head(15))


def main():
    # Load preprocessed features (X) from joblib
    logging.info("Loading preprocessed data from joblib...")
    X_train = load_data(config.TRAIN_X_PATH, "X_train")
    X_test = load_data(config.TEST_X_PATH, "X_test")

    # Load RAW (untransformed) target from cleaned_data.csv so MI is not
    # computed on Box-Cox values. MI does not require normality and the
    # raw target is more interpretable as a baseline reference.
    logging.info("Loading raw (untransformed) target from cleaned_data.csv for MI...")
    raw_df = pd.read_csv(config.CLEANED_CSV_PATH, usecols=[config.TARGET_COLUMN], low_memory=False)
    raw_y_all = pd.to_numeric(raw_df[config.TARGET_COLUMN], errors='coerce')

    # Align raw target rows to each split using the preserved integer indices
    y_train = raw_y_all.iloc[X_train.index].values
    y_test = raw_y_all.iloc[X_test.index].values

    X_whole = pd.concat([X_train, X_test], axis=0, ignore_index=True)
    y_whole = np.concatenate([y_train, y_test], axis=0)

    # Run analysis (Iterate over thresholds)
    for threshold in config.CLUSTERING_THRESHOLDS_TO_TEST:
        base_label = f"thresh_{threshold}" if threshold is not None else "original"
        logging.info(f"\n--- Processing MI for Threshold: {base_label} ---")

        # 1. Feature Selection (Always fit/select based on Train data to prevent leakage)
        if threshold is not None:
            report_path = os.path.join(config.BASE_RESULTS_DIR, f"cluster_membership_{base_label}.txt")

            if config.CLUSTERING_LINKAGE_METHOD == 'spearman':
                from clustering_utils import get_selected_features_spearman
                selected_feats = get_selected_features_spearman(X_train, threshold, save_report_path=report_path)
            elif config.CLUSTERING_LINKAGE_METHOD == 'spearman_ratio':
                from clustering_utils import get_selected_features_spearman_corr
                selected_feats = get_selected_features_spearman_corr(X_train, threshold, save_report_path=report_path)
            else:
                from clustering_utils import get_selected_features_by_clustering
                selected_feats = get_selected_features_by_clustering(
                    X_train, threshold, config.CLUSTERING_LINKAGE_METHOD, save_report_path=report_path
                )

            X_train_fs = X_train[selected_feats]
            X_whole_fs = X_whole[selected_feats]
            logging.info(f"  -> Features selected: {len(selected_feats)} / {X_train.shape[1]}")
            logging.info(f"  -> Cluster report saved to: {report_path}")
        else:
            X_train_fs = X_train
            X_whole_fs = X_whole
            logging.info(f"  -> Using all {X_train.shape[1]} features")

        # 2. Run MI for Train
        run_mutual_info_analysis(X_train_fs, y_train, base_label + "_train")

        # 3. Run MI for Whole Dataset
        run_mutual_info_analysis(X_whole_fs, y_whole, base_label + "_whole")


if __name__ == "__main__":
    main()