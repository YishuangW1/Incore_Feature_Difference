# Feature Importance Damage Modeling Pipeline

This repository contains the machine learning pipeline used for modeling and predicting structural damage probabilities difference between the simulated and the truth (Ranked Probability Skill Score - RPSS) for tornadoes. 

The pipeline is designed to systematically process older Unreinforced masonry (URM) building attributes, resolve multicollinearity using hierarchical feature clustering, evaluate feature importance via Mutual Information and Model Class Reliance (MCR), and ultimately achieve the importance
features that influence the descrepancy between the model simulation and field observations

## Pipeline Orchestration

The entire workflow is executed sequentially via the main orchestrator script:

### `main.py`
The master script that runs the entire end-to-end data analysis pipeline. It initializes the logging system (`pipeline.log`).

---

## Pipeline Scripts

These are the scripts currently enabled and executed sequentially by `main.py`:

### 1. `1_dataCleaning.py`
**Purpose:** Initial data ingestion and sanitization.
* Loads the raw dataset defined in `config.py`.
* Filters out explicitly excluded columns (e.g., metadata, images, identifiers).
* Handles initial missing values and forces correct data typing.
* Injects a `random_feature` column for baseline importance testing.
* Saves the sanitized data to `cleaned_data.csv`.

### 2. `2_dataPreprocessing_test.py`
**Purpose:** Data transformation and Train/Test splitting.
* **Target Processing:** Identifies rare target classes (count < 2) and drops them to ensure valid stratified sampling during train-test splitting.
* Performs an 80/20 train-test split (`train_test_split`) using stratified sampling.
* Applies Box-Cox transformation to the target variable to normalize its distribution (calculating parameters on the Train set and applying to the Test set to prevent data leakage).
* Encodes categorical variables (One-Hot or Ordinal) and scales numerical variables (StandardScaler).
* Saves the processed matrices as serialized `.pkl` files (`X_train_processed.pkl`, `y_train.pkl`, etc.).

### 3. `3_mutual_info.py`
**Purpose:** Non-linear feature importance analysis.
* Evaluates the Mutual Information (MI) between the processed features and the target variable to capture both linear and non-linear dependencies.
* Integrates with `clustering_utils.py` to evaluate features *after* handling multicollinearity.
* Generates publication-ready visualizations (e.g., `feature_importance_mi_top_10...png`) plotting the most significant features against a random baseline.

### 4. `4_modeling.py`
**Purpose:** Core machine learning modeling and evaluation.
* Dynamically clusters highly correlated features (based on the distance threshold in `config.py`) and selects one representative feature per cluster to eliminate collinearity.
* Trains baseline and advanced models (e.g., Decision Trees, Random Forests) on the reduced feature set.
* Evaluates model performance using standard metrics (R², RMSE, MAE).
* Saves evaluation metrics and identifies the best performing reference model.

### 5. `5_calculate_mcr.py`
**Purpose:** Model Class Reliance (MCR) calculation.
* Computes MCR by permuting features and observing the degradation in model performance, yielding an interval $[MCR_{-}, MCR_{+}]$ rather than a single point estimate.
* Uses the previously generated mapping dictionary to translate technical feature names (e.g., `cat__roof_shape_u`) into human-readable labels.
* Generates high-quality summary plots mapping feature importance ranges (`mcr_range_plot.png`).

### 6. `8_generate_report.py`
**Purpose:** Final result compilation.
* Aggregates the generated plots, evaluation metrics, and logs into a final cohesive visual report or summary document for review.

---

## Shared Dependencies & Utilities

While not directly executed by `main.py`, these files are imported and utilized extensively across the pipeline:

### `config.py`
The global configuration file storing the path and parameters. It defines:
* Input/Output file paths.
* Machine Learning hyperparameters (random state, test size, tree depth, n_estimators).
* Execution toggles (e.g., dropping infinite values, enabling RFE, selecting imputation strategy).
* Aesthetic settings for matplotlib/seaborn plots.

### `clustering_utils.py`
A core mathematical utility module.
* Implements **Hierarchical Agglomerative Clustering** using Spearman rank-order correlations.
* Contains `get_selected_features_by_clustering()`, a vital function that groups collinear variables based on a distance threshold and selects the most relevant representative feature to pass on to the ML models and Mutual Information evaluation.