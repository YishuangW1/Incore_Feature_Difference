# config
import os
import numpy as np
# --- Regressors ---
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb

# --- GENERAL ---
import os

RANDOM_STATE = int(os.getenv('PIPELINE_SEED', 42))

# ==========================================
#  PROBLEM TYPE
# ==========================================
PROBLEM_TYPE = 'regression'

# ==========================================
#  GEOSPATIAL CV SETTINGS
USE_GEOSPATIAL_CV = False
N_SPLITS_CV = 5

# ==========================================
# --- Infinite Value Handling ---
# ==========================================
DROP_INF = True  # Switch: True = drop rows with inf/-inf; False = replace them
INF_REPLACEMENT_VALUE = -100  # Replacement value: Used when DROP_INF = False

# ==========================================
# --- BoxCox & Encoding Target Handling ---
# ==========================================
# [REMOVED] Hardcoded Box-Cox parameters are now dynamically handled and saved to boxcox_params.json by 2_dataPreprocessing.py
ENCODING_METHOD = 'ordinal'  # onehot or ordinal
# Missing Value Strategy
# Strategy 1: 'drop_rows' -> Drop any row containing at least one NaN (No imputation)
# Strategy 2: 'impute' -> Impute missing values using SimpleImputer (only fitted on training data)
MISSING_VALUE_STRATEGY = 'impute'

# If Strategy is 'impute', choose strategy for numerical vs categorical
NUMERICAL_IMPUTE_STRATEGY = 'median'  # Options: 'mean', 'median', 'most_frequent'
CATEGORICAL_IMPUTE_STRATEGY = 'constant'  # 'constant' uses 'un_value'; otherwise 'most_frequent'

# ==========================================
#  HAZARD VARIABLE CONTROL (NEW!)
# ==========================================
# True: Hazard variables (EF, Lat, Lon, Archetype) enter main model training (participate in RFE, etc.).
# False: Force remove these variables during main model training (but Residual Analysis can still access them).
KEEP_HAZARD_VARIABLES = False

# HAZARD_COLUMNS = ['EF_scale', 'archetype','latitude', 'longitude' ]
HAZARD_COLUMNS = ['latitude', 'longitude']
#
# --- PATHS ---
DATA_DIR = 'processed_ml_data'
# New: Full data path specifically for Residual Analysis
RESIDUAL_DATA_DIR = 'processed_residual_data'

BASE_RESULTS_DIR = 'clustering_performance_results'
SHAP_RESULTS_DIR = 'shap_results_top_performers'
REPORT_DIR = 'reports'
INPUT_CSV_PATH = 'quad_nash_merged_rpss.csv'
CLEANED_CSV_PATH = 'cleaned_data.csv'
PIPELINE_LOG_PATH = 'pipeline.log'
REPORT_FILENAME = 'pipeline_visual_report.pdf'
DETAILED_RESULTS_CSV = os.path.join(BASE_RESULTS_DIR, 'clustering_performance_detailed_results.csv')
BEST_ESTIMATORS_PATH = os.path.join(BASE_RESULTS_DIR, 'best_estimators_per_combo.pkl')
MCR_SUMMARY_CSV = os.path.join(BASE_RESULTS_DIR, 'mcr_summary.csv')
MCR_RAW_VALUES_CSV = os.path.join(BASE_RESULTS_DIR, 'mcr_raw_values_all_models.csv')
MCR_CRITERION = 'r2'  # Options: 'rmse', 'r2'
MCR_R2_TOLERANCE = 0.01  # Tolerance for R2 (e.g., 0.01 means R2 >= Best_R2 - 0.01)

# Main Pipeline Paths
TRAIN_X_PATH = os.path.join(DATA_DIR, 'X_train_processed.pkl')
TRAIN_Y_PATH = os.path.join(DATA_DIR, 'y_train.pkl')
TEST_X_PATH = os.path.join(DATA_DIR, 'X_test_processed.pkl')
Y_TEST_PATH = os.path.join(DATA_DIR, 'y_test.pkl')
PREPROCESSOR_PATH = os.path.join(DATA_DIR, 'preprocessor.pkl')
BOXCOX_PARAMS_PATH = os.path.join(DATA_DIR, 'boxcox_params.json')  # Fitted lambda + offset saved here

N_REPEATS = 5

# --- DATA CLEANING ---
TARGET_COLUMN_FOR_NAN_DROP = 'RPSS_vs_perfect'
LOW_VARIATION_THRESHOLD = 1
KEYWORDS_TO_DROP = ['Unnamed', 'photos', 'details', 'complete_address', 'building_name_listing',
                    'building_name_current', '_damage']

# Note: Removed EF_scale, latitude, longitude, archetype here because they are now controlled by KEEP_HAZARD_VARIABLES
SPECIFIC_COLUMNS_TO_DROP = [
    'completed_by', 'damage_status', 'ref# (DELETE LATER)', 'complete_address',
    'building_name_listing', 'building_name_current', 'notes', 'tornado_name',
    'tornado_start_lat', 'tornado_start_long', 'tornado_end_lat',
    'tornado_end_long', 'national_register_listing_year', 'town',
    'located_in_historic_district', 'hazards_present_u', 'RPSS_vs_uniform',
    'building_count_in_group', 'perfect_RPS', 'building_count', 'perfect_RPS', 'uniform_RPS', 'incore_RPS'
    # 'RPSS_vs_perfect','incore_RPS'
]

COLUMNS_FOR_VALUE_REPLACEMENT = {
    'wall_thickness': {'un': '', 'not_applicable': 0},
    'overhang_length_u': {'un': '', 'not_applicable': 0},
    'parapet_height_m': {'un': '', 'not_applicable': 0}
}

# --- PREPROCESSING & MODELING CONFIGURATION ---
TARGET_COLUMN = 'RPSS_vs_perfect'
TEST_SIZE = 0.2
PERFORM_RFE = False
N_FEATURES_TO_SELECT = 19
PERMUTATION_IMPORTANCE_REPEATS = 1
MI_TOP_N = 10  # Number of top features to plot in mutual info evaluation

# --- PERMUTATION IMPORTANCE CONFIGURATION ---
NUM_MODEL_RUNS_FOR_PERMUTATION = 25
NUM_PERMUTATION_REPEATS = 30

KEYWORDS_TO_REMOVE_FROM_X = [
    'demolishing_year', 'demoshed_by_2023', 'buidling_use_after_tornado',
    'buidling_use_plan_after_tornado', 'simulated_damage', 'estimated',
    'damage', 'status_u', 'exist', 'demolish', 'failure', 'after'
]

# Clustering Thresholds
# If list is empty, clustering is skipped.
# CLUSTERING_THRESHOLDS_TO_TEST = [380]
# CLUSTERING_LINKAGE_METHOD = 'spearman'
CLUSTERING_THRESHOLDS_TO_TEST = [0.675]
CLUSTERING_LINKAGE_METHOD = 'spearman_ratio'
PERFORMANCE_THRESHOLD_FOR_PLOT = 0

# ==========================================
#  REGRESSION CONFIGURATION
# =======================================s===

# --- Regression Specific Settings ---
APPLY_TARGET_TRANSFORMATION = True  # Regression logic
TARGET_TRANSFORMATION_METHOD = 'boxcox'
BALANCING_METHOD = None  # No SMOTE for regression
GRIDSEARCH_SCORING_METRIC = 'neg_mean_squared_error'

MODELS_TO_BENCHMARK = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(random_state=RANDOM_STATE),
    "Lasso": Lasso(random_state=RANDOM_STATE),
    "ElasticNet": ElasticNet(random_state=RANDOM_STATE),
    "SVR": SVR(),
    "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
    "Random Forest": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    "Hist Gradient Boosting": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
    "XGBoost": xgb.XGBRegressor(random_state=RANDOM_STATE),
    "LightGBM": lgb.LGBMRegressor(random_state=RANDOM_STATE, verbosity=-1),
}

PARAM_GRIDS = {
    # Ridge/Lasso: 处理 VIF 的第一道防线
    "Ridge": {'alpha': [1.0, 10.0, 50.0, 100.0, 200.0]},
    "Lasso": {'alpha': [0.01, 0.1, 1.0, 5.0, 10.0]},
    "ElasticNet": {
        'alpha': [0.1, 1.0, 10.0, 20.0],
        'l1_ratio': [0.5, 0.8, 0.9]
    },

    # SVR: 小样本下表现稳健
    "SVR": {'kernel': ['linear', 'rbf'], 'C': [0.1, 1, 10], 'epsilon': [0.05, 0.1, 0.2]},

    # Tree Models: 根据边界命中情况向上微调
    "Decision Tree": {'max_depth': [3, 4, 6, 8], 'min_samples_leaf': [2, 5, 10]},
    "Random Forest": {
        'n_estimators': [200, 300],
        'max_depth': [4, 5, 6],
        'min_samples_leaf': [5, 8],
        'max_features': ['sqrt']
    },

    # Boosting: 增加学习率并允许深一点的树，同时维持采样率
    "Gradient Boosting": {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 4, 5],
        'min_samples_leaf': [5, 10],
        'subsample': [0.8, 0.9]
    },
    "Hist Gradient Boosting": {
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 4, 5],
        'min_samples_leaf': [5, 10],
        'l2_regularization': [0.01, 0.1, 1.0]
    },
    "XGBoost": {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 4, 5],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9],
        'reg_alpha': [0.1, 1.0],
        'reg_lambda': [1.0, 5.0]
    },
    "LightGBM": {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'num_leaves': [15, 31],
        'min_child_samples': [5, 10],
        'feature_fraction': [0.8, 0.9],
        'lambda_l1': [0.1, 1.0]
    },
}

# --- VISUALIZATION ---
VISUALIZATION = {
    'main_palette': 'viridis',
    'diverging_palette': 'coolwarm',
    'plot_style': 'seaborn-v0_8-white'
}