import pandas as pd
import numpy as np
import warnings
from dython.nominal import associations
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

def get_selected_features_by_clustering(original_df, distance_thresh, linkage_meth, save_report_path=None):
    """
    Performs hierarchical clustering on features to remove collinearity.
    Uses Cramer's V (via dython) to calculate correlation/association matrix.
    
    Args:
        original_df (pd.DataFrame): DataFrame containing features to cluster.
        distance_thresh (float): Threshold for cutting the dendrogram. 
        linkage_meth (str): Linkage method (e.g., 'complete').
        save_report_path (str, optional): Path to save the text report.
        
    Returns:
        list: List of selected feature names.
    """
    if distance_thresh is None or pd.isna(distance_thresh):
        return original_df.columns.tolist()

    feature_names = original_df.columns.tolist()
    
    # Identify categorical columns to guarantee dython uses Cramer's V and Correlation Ratio
    nominal_cols = [c for c in feature_names if c.startswith('cat__')]
    
    # Calculate Association/Correlation Matrix
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Dython will automatically use Pearson for Num-Num, Cramer's V for Nom-Nom, and Correlation Ratio for Mixed
        assoc_result = associations(
            original_df, 
            nominal_columns=nominal_cols,
            nom_nom_assoc='cramer', 
            compute_only=True
        )
        assoc_df = assoc_result['corr'].fillna(0)

    # Convert Correlation to Distance (1 - |Corr|)
    distance_mat = 1 - np.abs(assoc_df.values)
    np.fill_diagonal(distance_mat, 0)
    
    # Hierarchical Clustering
    condensed_dist_mat = squareform(distance_mat, checks=False)
    linked = hierarchy.linkage(condensed_dist_mat, method=linkage_meth)
    cluster_labels = hierarchy.fcluster(linked, t=distance_thresh, criterion='distance')

    # Select Representatives & Generate Report
    selected_features = []
    report_lines = []
    
    unique_labels = sorted(list(set(cluster_labels)))
    report_lines.append(f"--- Clustering Report (Threshold={distance_thresh}) ---")
    report_lines.append(f"Total Features: {len(feature_names)}")
    report_lines.append(f"Total Clusters: {len(unique_labels)}\n")

    for i in unique_labels:
        cluster_indices = [idx for idx, label in enumerate(cluster_labels) if label == i]
        if not cluster_indices: continue
        
        members = [feature_names[idx] for idx in cluster_indices]
        
        if len(cluster_indices) == 1:
            representative = members[0]
            selected_features.append(representative)
            report_lines.append(f"Cluster {i} (1 feature): {representative} [KEPT]")
        else:
            # Pick feature with highest sum of absolute correlations within the cluster
            sum_abs_assoc = np.abs(assoc_df.iloc[cluster_indices, cluster_indices].values).sum(axis=1)
            representative_index = np.argmax(sum_abs_assoc)
            representative = members[representative_index]
            selected_features.append(representative)
            
            report_lines.append(f"Cluster {i} ({len(members)} features):")
            report_lines.append(f"  Representative: {representative} [KEPT]")
            report_lines.append(f"  Members: {', '.join(members)}")
        
        report_lines.append("") # Blank line

    # Save Report if path provided
    if save_report_path:
        try:
            with open(save_report_path, 'w') as f:
                f.write('\n'.join(report_lines))
            # print(f"Clustering report saved to {save_report_path}") 
        except Exception as e:
            print(f"Failed to save clustering report: {e}")

    return sorted(list(set(selected_features)))

def calculate_cluster_metrics(original_df, distance_thresh, linkage_meth):
    """
    Calculates intrinsic clustering metrics (Silhouette Score).
    Returns a dict with 'silhouette_score' and 'n_clusters'.
    """
    if distance_thresh is None or pd.isna(distance_thresh):
        return {'silhouette_score': None, 'n_clusters': len(original_df.columns)}

    # Calculate Association/Correlation Matrix
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assoc_result = associations(original_df, nom_nom_assoc='cramer', compute_only=True)
        assoc_df = assoc_result['corr'].fillna(0)

    # Convert to Distance
    distance_mat = 1 - np.abs(assoc_df.values)
    np.fill_diagonal(distance_mat, 0)
    condensed_dist_mat = squareform(distance_mat, checks=False)

    # Clustering
    linked = hierarchy.linkage(condensed_dist_mat, method=linkage_meth)
    cluster_labels = hierarchy.fcluster(linked, t=distance_thresh, criterion='distance')
    
    n_clusters = len(set(cluster_labels))
    
    if n_clusters < 2 or n_clusters >= len(original_df.columns):
        return {'silhouette_score': -1, 'n_clusters': n_clusters} # Undefined for 1 cluster or singletons
        
    from sklearn.metrics import silhouette_score
    score = silhouette_score(distance_mat, cluster_labels, metric='precomputed')
    
    return {'silhouette_score': score, 'n_clusters': n_clusters}

def get_selected_features_ward(original_df, distance_thresh, save_report_path=None):
    """
    Performs hierarchical clustering using Ward's Linkage on Euclidean Distance.
    This allows for thresholds > 1 (e.g., 1.5, 2.0, etc.).
    """
    if distance_thresh is None:
        return original_df.columns.tolist()

    feature_names = original_df.columns.tolist()
    
    # 1. Transpose: Features become rows
    X_features = original_df.T
    
    # 2. Ward Linkage (Euclidean is implicit/required for Ward)
    # Z contains the dendrogram
    Z = hierarchy.linkage(X_features, method='ward', metric='euclidean')
    
    # 3. Cut Dendrogram
    cluster_labels = hierarchy.fcluster(Z, t=distance_thresh, criterion='distance')
    
    # 4. Select Representatives (Centrality: Closest to Cluster Center)
    selected_features = []
    report_lines = []
    
    unique_labels = sorted(list(set(cluster_labels)))
    report_lines.append(f"--- Ward Clustering Report (Threshold={distance_thresh}) ---")
    report_lines.append(f"Total Features: {len(feature_names)}")
    report_lines.append(f"Total Clusters: {len(unique_labels)}\n")

    from scipy.spatial.distance import euclidean

    for i in unique_labels:
        cluster_indices = [idx for idx, label in enumerate(cluster_labels) if label == i]
        members = [feature_names[idx] for idx in cluster_indices]
        
        if len(members) == 1:
            representative = members[0]
            selected_features.append(representative)
            report_lines.append(f"Cluster {i} (1 feature): {representative} [KEPT]")
        else:
            # Calculate Centroid of the cluster
            cluster_data = X_features.iloc[cluster_indices]
            centroid = cluster_data.mean(axis=0)
            
            # Find feature closest to centroid (Euclidean)
            distances = cluster_data.apply(lambda row: euclidean(row, centroid), axis=1)
            representative = distances.idxmin() # Index is feature name
            
            selected_features.append(representative)
            report_lines.append(f"Cluster {i} ({len(members)} features):")
            report_lines.append(f"  Representative: {representative} [KEPT]")
            report_lines.append(f"  Members: {', '.join(members)}")
            
        report_lines.append("")

    if save_report_path:
        with open(save_report_path, 'w') as f:
            f.write('\n'.join(report_lines))

    return sorted(list(set(selected_features)))


def get_selected_features_spearman_corr(original_df, correlation_thresh, linkage_meth='complete', save_report_path=None):
    """
    Performs clustering using the EXACT Spearman correlation coefficient matrix directly.
    Distance is defined as 1 - |Spearman correlation|.
    So a correlation_thresh of 0.85 means we merge things with |correlation| > 0.85 
    (passed into hierarchy.fcluster as threshold 0.15).
    """
    if correlation_thresh is None:
        return original_df.columns.tolist()

    feature_names = original_df.columns.tolist()
    
    # 1. Exact Spearman correlation matrix via pandas
    corr_matrix = original_df.corr(method='spearman').fillna(0)
    
    # 2. Convert Correlation to Distance (1 - |Corr|)
    distance_mat = 1 - np.abs(corr_matrix.values)
    np.fill_diagonal(distance_mat, 0)
    
    # 3. Hierarchical Clustering
    condensed_dist_mat = squareform(distance_mat, checks=False)
    
    linked = hierarchy.linkage(condensed_dist_mat, method=linkage_meth)
    
    # Note: the input threshold is correlation (e.g. 0.85), so distance threshold is 1 - 0.85 = 0.15
    dist_t = 1.0 - correlation_thresh
    cluster_labels = hierarchy.fcluster(linked, t=dist_t, criterion='distance')

    # 4. Select Representatives
    selected_features = []
    report_lines = []

    unique_labels = sorted(list(set(cluster_labels)))
    report_lines.append(f"--- Exact Spearman Correlation Report (|Rho| Thresh={correlation_thresh}) ---")
    report_lines.append(f"Distance Threshold used: {dist_t:.4f}")
    report_lines.append(f"Total Features: {len(feature_names)}")
    report_lines.append(f"Total Clusters: {len(unique_labels)}\n")

    for i in unique_labels:
        cluster_indices = [idx for idx, label in enumerate(cluster_labels) if label == i]
        members = [feature_names[idx] for idx in cluster_indices]

        if len(members) == 1:
            representative = members[0]
            selected_features.append(representative)
            report_lines.append(f"Cluster {i} (1 feature): {representative} [KEPT]")
        else:
            sub_corr = np.abs(corr_matrix.iloc[cluster_indices, cluster_indices].values)
            representative_index = np.argmax(sub_corr.sum(axis=1))
            representative = members[representative_index]

            selected_features.append(representative)
            report_lines.append(f"Cluster {i} ({len(members)} features):")
            report_lines.append(f"  Representative: {representative} [KEPT]")
            report_lines.append(f"  Members: {', '.join(members)}")

        report_lines.append("")

    if save_report_path:
        with open(save_report_path, 'w') as f:
            f.write('\n'.join(report_lines))

    return sorted(list(set(selected_features)))



def get_selected_features_spearman(original_df, distance_thresh, save_report_path=None):
    """
    Performs hierarchical clustering using Ward's Linkage on Spearman-ranked data.

    Implementation:
      1. Convert features to ranks (Spearman rank transform).
      2. Transpose so each feature is a point in N-dimensional rank space.
      3. Run Ward's Linkage using Euclidean distance on the rank vectors.
         This clusters features by monotonic rank relationships.
      4. Cut the dendrogram at 'distance_thresh' (Euclidean units in rank space).

    Representative selection: feature whose rank vector is closest to the cluster
    centroid (most average behavior within the cluster in rank space).
    """
    if distance_thresh is None:
        return original_df.columns.tolist()

    feature_names = original_df.columns.tolist()

    # 1. Rank Transform (Spearman)
    X_ranks = original_df.rank(axis=0)

    # 2. Transpose for Feature Clustering (features become rows)
    X_features = X_ranks.T

    # 3. Ward Linkage on Ranks (Euclidean distance in rank space)
    Z = hierarchy.linkage(X_features, method='ward', metric='euclidean')

    # 4. Cut Dendrogram
    cluster_labels = hierarchy.fcluster(Z, t=distance_thresh, criterion='distance')

    # 5. Select Representatives
    selected_features = []
    report_lines = []

    unique_labels = sorted(list(set(cluster_labels)))
    report_lines.append(f"--- Spearman (Ward on Ranks) Report (Threshold={distance_thresh}) ---")
    report_lines.append(f"Total Features: {len(feature_names)}")
    report_lines.append(f"Total Clusters: {len(unique_labels)}\n")

    from scipy.spatial.distance import euclidean

    for i in unique_labels:
        cluster_indices = [idx for idx, label in enumerate(cluster_labels) if label == i]
        members = [feature_names[idx] for idx in cluster_indices]

        if len(members) == 1:
            representative = members[0]
            selected_features.append(representative)
            report_lines.append(f"Cluster {i} (1 feature): {representative} [KEPT]")
        else:
            # Centroid in Rank Space
            cluster_data = X_features.iloc[cluster_indices]
            centroid = cluster_data.mean(axis=0)

            # Feature closest to centroid (most average rank behavior)
            distances = cluster_data.apply(lambda row: euclidean(row, centroid), axis=1)
            representative = distances.idxmin()

            selected_features.append(representative)
            report_lines.append(f"Cluster {i} ({len(members)} features):")
            report_lines.append(f"  Representative: {representative} [KEPT]")
            report_lines.append(f"  Members: {', '.join(members)}")

        report_lines.append("")

    if save_report_path:
        with open(save_report_path, 'w') as f:
            f.write('\n'.join(report_lines))

    return sorted(list(set(selected_features)))


def get_selected_features_spearman_corr(original_df, correlation_thresh, linkage_meth='complete', save_report_path=None):
    """
    Performs clustering using the EXACT Spearman correlation coefficient matrix directly.
    Distance is defined as 1 - |Spearman correlation|.
    So a correlation_thresh of 0.85 means we merge things with |correlation| > 0.85 
    (passed into hierarchy.fcluster as threshold 0.15).
    """
    if correlation_thresh is None:
        return original_df.columns.tolist()

    feature_names = original_df.columns.tolist()
    
    # 1. Exact Spearman correlation matrix via pandas
    corr_matrix = original_df.corr(method='spearman').fillna(0)
    
    # 2. Convert Correlation to Distance (1 - |Corr|)
    distance_mat = 1 - np.abs(corr_matrix.values)
    np.fill_diagonal(distance_mat, 0)
    
    # 3. Hierarchical Clustering
    condensed_dist_mat = squareform(distance_mat, checks=False)
    
    linked = hierarchy.linkage(condensed_dist_mat, method=linkage_meth)
    
    # Note: the input threshold is correlation (e.g. 0.85), so distance threshold is 1 - 0.85 = 0.15
    dist_t = 1.0 - correlation_thresh
    cluster_labels = hierarchy.fcluster(linked, t=dist_t, criterion='distance')

    # 4. Select Representatives
    selected_features = []
    report_lines = []

    unique_labels = sorted(list(set(cluster_labels)))
    report_lines.append(f"--- Exact Spearman Correlation Report (|Rho| Thresh={correlation_thresh}) ---")
    report_lines.append(f"Distance Threshold used: {dist_t:.4f}")
    report_lines.append(f"Total Features: {len(feature_names)}")
    report_lines.append(f"Total Clusters: {len(unique_labels)}\n")

    for i in unique_labels:
        cluster_indices = [idx for idx, label in enumerate(cluster_labels) if label == i]
        members = [feature_names[idx] for idx in cluster_indices]

        if len(members) == 1:
            representative = members[0]
            selected_features.append(representative)
            report_lines.append(f"Cluster {i} (1 feature): {representative} [KEPT]")
        else:
            sub_corr = np.abs(corr_matrix.iloc[cluster_indices, cluster_indices].values)
            representative_index = np.argmax(sub_corr.sum(axis=1))
            representative = members[representative_index]

            selected_features.append(representative)
            report_lines.append(f"Cluster {i} ({len(members)} features):")
            report_lines.append(f"  Representative: {representative} [KEPT]")
            report_lines.append(f"  Members: {', '.join(members)}")

        report_lines.append("")

    if save_report_path:
        with open(save_report_path, 'w') as f:
            f.write('\n'.join(report_lines))

    return sorted(list(set(selected_features)))



