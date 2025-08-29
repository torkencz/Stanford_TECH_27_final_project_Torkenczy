#!/usr/bin/env python3
"""
Unified ML Pipeline for PBMC Single-Cell Classification Project
Works with EDA results from both annotated and not_annotated approaches
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                            precision_recall_fscore_support, roc_auc_score, average_precision_score,
                            roc_curve, precision_recall_curve)
from sklearn.preprocessing import label_binarize
import os
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Global verbose flag
VERBOSE = True

def vprint(*args, **kwargs):
    """Verbose print - only prints if VERBOSE is True"""
    if VERBOSE:
        print(*args, **kwargs)

def set_verbose(verbose):
    """Set global verbose flag"""
    global VERBOSE
    VERBOSE = verbose

# Keras/TensorFlow imports for deep learning models
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
    from tensorflow.keras.optimizers import Adam
    KERAS_AVAILABLE = True
    # Suppress TensorFlow warnings
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
except ImportError:
    vprint("TensorFlow/Keras not available. Neural network models will be skipped.")
    KERAS_AVAILABLE = False

# Set high-quality plotting parameters
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

def load_processed_data(approach):
    """Load processed data from EDA results"""
    vprint(f"Loading processed {approach} data...")
    
    data_dir = f'data/{approach}'
    processed_files = [f for f in os.listdir(data_dir) if f.endswith('_processed.h5ad')]
    
    if not processed_files:
        raise FileNotFoundError(f"No processed .h5ad files found in {data_dir}")
    
    # Filter files to avoid duplicates - prefer standard versions over _full variants
    # For annotated: use pbmc3k_processed.h5ad and pbmc_multiome_full_processed.h5ad (both have 2000 HVGs)
    # For not_annotated: use standard versions
    filtered_files = []
    if approach == 'annotated':
        # Use consistent HVG versions (2000 genes each)
        priority_files = ['pbmc3k_processed.h5ad', 'pbmc_multiome_full_processed.h5ad']
        for pfile in priority_files:
            if pfile in processed_files:
                filtered_files.append(pfile)
    else:
        # For not_annotated, exclude _full variants to avoid duplicates
        filtered_files = [f for f in processed_files if '_full_processed.h5ad' not in f]
    
    if not filtered_files:
        vprint(f"No suitable processed files found, using all available")
        filtered_files = processed_files
    
    datasets = {}
    for file in filtered_files:
        dataset_name = file.replace('_processed.h5ad', '')
        file_path = os.path.join(data_dir, file)
        
        vprint(f"Loading {dataset_name} from {file_path}")
        adata = sc.read_h5ad(file_path)
        
        vprint(f"{dataset_name}: {adata.n_obs} cells × {adata.n_vars} genes")
        vprint(f"Cell types: {adata.obs['cell_type'].nunique()}")
        
        datasets[dataset_name] = adata
    
    return datasets

def create_keras_mlp(input_dim, n_classes):
    """Create Keras MLP model (from Class 5 - Feed-Forward Neural Network)"""
    if not KERAS_AVAILABLE:
        return None
    
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.1),
        Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def create_keras_cnn(input_dim, n_classes):
    """Create Keras 1D CNN model (from Class 6 - Convolutional Neural Network)"""
    if not KERAS_AVAILABLE:
        return None
    
    model = Sequential([
        tf.keras.layers.Reshape((input_dim, 1), input_shape=(input_dim,)),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=32, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

def preprocess_for_ml(adata, min_genes=200, min_cells=3, n_top_genes=2000):
    """Preprocess AnnData for machine learning"""
    vprint(f"Preprocessing {adata.n_obs} cells × {adata.n_vars} genes...")
    
    # Filter cells and genes
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    
    # Remove cells with no cell type annotation
    adata = adata[~adata.obs['cell_type'].isna()].copy()
    adata = adata[adata.obs['cell_type'] != 'filtered'].copy()  # Remove filtered cells if present
    
    # Remove cells with Unknown cell type annotation
    original_cells = adata.n_obs
    adata = adata[adata.obs['cell_type'] != 'Unknown'].copy()
    unknown_filtered = original_cells - adata.n_obs
    if unknown_filtered > 0:
        vprint(f"Filtered out {unknown_filtered} Unknown cells ({unknown_filtered/original_cells*100:.1f}%)")
    
    vprint(f"After filtering: {adata.n_obs} cells × {adata.n_vars} genes")
    
    # Check if data is already normalized (from EDA processing)
    if hasattr(adata, 'raw') and adata.raw is not None:
        vprint("Data already processed in EDA - using existing normalization")
        # Data is already normalized and log-transformed from EDA
        # Just apply HVG filtering and scaling
        
        # Find highly variable genes
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=n_top_genes)
        adata = adata[:, adata.var.highly_variable]
        
        vprint(f"Highly variable genes: {adata.n_vars}")
        
        # Scale data
        sc.pp.scale(adata, max_value=10)
    else:
        vprint("Applying fresh normalization (data not pre-processed)")
        # Store raw data
        adata.raw = adata
        
        # Normalize and log transform
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        
        # Find highly variable genes
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=n_top_genes)
        adata = adata[:, adata.var.highly_variable]
        
        vprint(f"Highly variable genes: {adata.n_vars}")
        
        # Scale data
        sc.pp.scale(adata, max_value=10)
    
    return adata

def run_transfer_learning_knn(X_reference, y_reference, genes_reference, X_query, genes_query, k=15):
    """
    Run k-NN based transfer learning using reference dataset
    
    This is an approach that uses a reference dataset to annotate query cells
    based on gene expression similarity. It finds common genes between datasets and uses
    k-nearest neighbors to transfer cell type annotations.
    
    Parameters:
    -----------
    X_reference : array-like
        Reference dataset expression matrix (cells × genes)
    y_reference : array-like
        Reference dataset cell type labels
    genes_reference : array-like
        Reference dataset gene names
    X_query : array-like
        Query dataset expression matrix (cells × genes)
    genes_query : array-like
        Query dataset gene names
    k : int
        Number of neighbors for k-NN classifier
        
    Returns:
    --------
    predicted_labels : array-like or None
        Predicted cell type labels for query dataset
    confidence_scores : array-like or None
        Confidence scores for predictions
    """
    try:
        from scipy import sparse
        from sklearn.preprocessing import StandardScaler
        
        # Find common genes between reference and query
        ref_gene_set = set(genes_reference)
        query_gene_set = set(genes_query)
        common_genes = list(ref_gene_set.intersection(query_gene_set))
        
        vprint(f"Found {len(common_genes)} common genes for transfer learning")
        
        if len(common_genes) < 100:
            vprint("Too few common genes for reliable transfer learning")
            return None, None
        
        # Get indices for common genes
        ref_indices = [np.where(genes_reference == gene)[0][0] for gene in common_genes 
                      if gene in genes_reference]
        query_indices = [np.where(genes_query == gene)[0][0] for gene in common_genes 
                        if gene in genes_query]
        
        # Subset data to common genes
        X_ref_subset = X_reference[:, ref_indices]
        X_query_subset = X_query[:, query_indices]
        
        # Convert to dense if sparse
        if sparse.issparse(X_ref_subset):
            X_ref_subset = X_ref_subset.toarray()
        if sparse.issparse(X_query_subset):
            X_query_subset = X_query_subset.toarray()
        
        # Preprocessing for transfer learning - use RAW counts (no double normalization)
        def preprocess_transfer(X, use_raw_counts=True):
            # Handle NaN values first
            if np.any(np.isnan(X)):
                vprint(f"Found NaN values, replacing with zeros...")
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            if use_raw_counts:
                # Use raw counts for transfer learning (avoid double normalization)
                vprint(f"Using raw counts for transfer learning (avoiding double normalization)")
                # Simple standardization of raw counts
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Final NaN check after scaling
                if np.any(np.isnan(X_scaled)) or np.any(np.isinf(X_scaled)):
                    vprint(f"Found NaN/inf after scaling, cleaning...")
                    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                # Fallback: normalize and log transform (for unprocessed data)
                vprint(f"Applying normalization and log transform")
                cell_totals = X.sum(axis=1)
                cell_totals[cell_totals == 0] = 1
                X_norm = (X / cell_totals[:, np.newaxis]) * 10000
                X_log = np.log1p(X_norm)
                
                # Handle any remaining NaN/inf values after log transform
                if np.any(np.isnan(X_log)) or np.any(np.isinf(X_log)):
                    vprint(f"Found NaN/inf after log transform, cleaning...")
                    X_log = np.nan_to_num(X_log, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Scale
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_log)
                
                # Final NaN check after scaling
                if np.any(np.isnan(X_scaled)) or np.any(np.isinf(X_scaled)):
                    vprint(f"Found NaN/inf after scaling, final cleaning...")
                    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            
            return X_scaled, scaler
        
        # Determine if we should use raw counts (if data comes from adata.raw)
        use_raw_counts = True  # Default to raw counts to avoid double normalization
        
        X_ref_processed, ref_scaler = preprocess_transfer(X_ref_subset, use_raw_counts=use_raw_counts)
        
        # Process query data with same approach
        if np.any(np.isnan(X_query_subset)):
            vprint(f"Found NaN values in query data, replacing with zeros...")
            X_query_subset = np.nan_to_num(X_query_subset, nan=0.0, posinf=0.0, neginf=0.0)
        
        if use_raw_counts:
            # Simple standardization for raw counts
            X_query_processed = ref_scaler.transform(X_query_subset)
        else:
            # Normalize and log transform query data (fallback)
            query_cell_totals = X_query_subset.sum(axis=1)[:, np.newaxis] + 1e-10
            X_query_norm = (X_query_subset / query_cell_totals) * 10000
            X_query_log = np.log1p(X_query_norm)
            
            # Handle NaN/inf after log transform
            if np.any(np.isnan(X_query_log)) or np.any(np.isinf(X_query_log)):
                vprint(f"Found NaN/inf in query after log transform, cleaning...")
                X_query_log = np.nan_to_num(X_query_log, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Apply scaler
            X_query_processed = ref_scaler.transform(X_query_log)
        
        # Final check
        if np.any(np.isnan(X_query_processed)) or np.any(np.isinf(X_query_processed)):
            vprint(f"Found NaN/inf in query after final processing, cleaning...")
            X_query_processed = np.nan_to_num(X_query_processed, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Use k-NN for transfer learning
        knn_transfer = KNeighborsClassifier(n_neighbors=k, weights='distance')
        knn_transfer.fit(X_ref_processed, y_reference)
        
        # Predict on query data
        predicted_labels = knn_transfer.predict(X_query_processed)
        prediction_proba = knn_transfer.predict_proba(X_query_processed)
        confidence_scores = np.max(prediction_proba, axis=1)
        
        return predicted_labels, confidence_scores
        
    except Exception as e:
        vprint(f"Transfer learning failed: {e}")
        return None, None

def scanpy_ingest_transfer_learning(adata_reference, adata_query, use_rep='X_pca', k=50):
    """
    Apply Scanpy's sc.tl.ingest for transfer learning annotation mapping
    
    This is Scanpy's state-of-the-art method for mapping annotations from a reference
    dataset to a query dataset. It uses neighborhood structure and embedding to 
    transfer annotations with high accuracy.
    
    Parameters:
    -----------
    adata_reference : AnnData
        Reference dataset with cell type annotations
    adata_query : AnnData  
        Query dataset to annotate
    use_rep : str, default 'X_pca'
        Representation to use for mapping ('X_pca', 'X', or obsm key)
    k : int, default 50
        Number of neighbors for mapping
        
    Returns:
    --------
    predicted_labels : array-like or None
        Predicted cell type labels for query dataset
    confidence_scores : array-like or None
        Confidence scores for predictions (based on neighbor consensus)
    """
    try:
        vprint(f"Training Scanpy Ingest Transfer Learning...")
        
        # Make copies to avoid modifying original data
        adata_ref = adata_reference.copy()
        adata_query = adata_query.copy()
        
        # Ensure we have the required representation
        if use_rep == 'X_pca':
            # Compute PCA if not present
            if 'X_pca' not in adata_ref.obsm:
                vprint(f"Computing PCA for reference dataset...")
                sc.tl.pca(adata_ref, n_comps=50, svd_solver='arpack')
            if 'X_pca' not in adata_query.obsm:
                vprint(f"Computing PCA for query dataset...")
                sc.tl.pca(adata_query, n_comps=50, svd_solver='arpack')
        
        # Find common genes between datasets
        common_genes = adata_ref.var_names.intersection(adata_query.var_names)
        vprint(f"Found {len(common_genes)} common genes for ingest")
        
        if len(common_genes) < 100:
            vprint(f"Very few common genes ({len(common_genes)}), results may be unreliable")
        
        # Subset to common genes
        adata_ref_common = adata_ref[:, common_genes].copy()
        adata_query_common = adata_query[:, common_genes].copy()
        
        # Recompute PCA on common genes if using X_pca
        if use_rep == 'X_pca':
            vprint(f"Recomputing PCA on {len(common_genes)} common genes...")
            sc.tl.pca(adata_ref_common, n_comps=min(50, len(common_genes)-1), svd_solver='arpack')
            sc.tl.pca(adata_query_common, n_comps=min(50, len(common_genes)-1), svd_solver='arpack')
        
        # Compute neighbors for reference dataset
        vprint(f"Computing neighborhood graph for reference...")
        sc.pp.neighbors(adata_ref_common, n_neighbors=k, use_rep=use_rep, random_state=42)
        
        # Compute UMAP for reference dataset (required for ingest)
        vprint(f"Computing UMAP for reference dataset...")
        sc.tl.umap(adata_ref_common, random_state=42)
        
        # Apply ingest to map query to reference space
        vprint(f"Applying sc.tl.ingest to map query dataset...")
        sc.tl.ingest(adata_query_common, adata_ref_common, obs='cell_type')
        
        # Extract predicted labels
        predicted_labels = adata_query_common.obs['cell_type'].values
        
        # Calculate confidence scores based on ingest results
        # Use a simplified approach that's more robust
        if 'X_umap' in adata_query_common.obsm and 'X_umap' in adata_ref_common.obsm:
            # Calculate confidence based on distance to reference cells in UMAP space
            from sklearn.neighbors import NearestNeighbors
            
            query_umap = adata_query_common.obsm['X_umap']
            ref_umap = adata_ref_common.obsm['X_umap']
            ref_labels = adata_ref_common.obs['cell_type'].values
            
            # Find nearest neighbors in UMAP space
            nbrs = NearestNeighbors(n_neighbors=min(10, len(ref_labels)), algorithm='auto')
            nbrs.fit(ref_umap)
            distances, indices = nbrs.kneighbors(query_umap)
            
            confidence_scores = []
            for i, predicted_label in enumerate(predicted_labels):
                # Get labels of nearest neighbors
                neighbor_labels = ref_labels[indices[i]]
                # Calculate fraction of neighbors with same label as prediction
                matching_neighbors = np.sum(neighbor_labels == predicted_label)
                confidence = matching_neighbors / len(neighbor_labels)
                confidence_scores.append(confidence)
            
            confidence_scores = np.array(confidence_scores)
        else:
            # Fallback: uniform confidence based on label frequency
            from collections import Counter
            label_counts = Counter(predicted_labels)
            total_predictions = len(predicted_labels)
            confidence_scores = np.array([label_counts[label] / total_predictions for label in predicted_labels])
        
        vprint(f"Scanpy ingest completed successfully")
        vprint(f"Average confidence: {confidence_scores.mean():.3f}")
        
        return predicted_labels, confidence_scores
        
    except Exception as e:
        vprint(f"Scanpy ingest transfer learning failed: {e}")
        return None, None

def harmonize_cell_types(y_train, y_test):
    """Harmonize cell type labels between datasets for cross-dataset analysis"""
    
    # Define mapping from diverse labels to common categories
    label_mapping = {
        # T cells
        'Naive CD4 T': 'CD4+ T',
        'Memory CD4 T': 'CD4+ T', 
        'CD4 Naive': 'CD4+ T',
        'CD4 TCM': 'CD4+ T',
        'CD4 TEM': 'CD4+ T',
        'Treg': 'CD4+ T',
        
        'CD8 T': 'CD8+ T',
        'CD8 Naive': 'CD8+ T',
        'CD8 TEM_1': 'CD8+ T',
        'CD8 TEM_2': 'CD8+ T',
        
        # B cells
        'B': 'B',
        'Naive B': 'B',
        'Memory B': 'B', 
        'Intermediate B': 'B',
        'Plasma': 'B',
        
        # Monocytes
        'CD14+ Mono': 'Monocytes',
        'CD14 Mono': 'Monocytes',
        'FCGR3A+ Mono': 'Monocytes', 
        'CD16 Mono': 'Monocytes',
        
        # NK cells
        'NK': 'NK',
        
        # Dendritic cells
        'DC': 'DC',
        'cDC': 'DC',
        'pDC': 'DC',
        
        # Other
        'Platelet': 'Other',
        'HSPC': 'Other',
        'MAIT': 'Other',
        'gdT': 'Other',
    }
    
    # Apply mapping
    y_train_mapped = np.array([label_mapping.get(label, 'Other') for label in y_train])
    y_test_mapped = np.array([label_mapping.get(label, 'Other') for label in y_test])
    
    return y_train_mapped, y_test_mapped

def run_ml_analysis(train_adata, test_adata, output_dir, n_components=50, include_transfer_learning=False, 
                   train_adata_orig=None, test_adata_orig=None):
    """Run comprehensive ML analysis"""
    vprint(f"Running Machine Learning Analysis")
    vprint("="* 50)
    
    # Prepare training data
    X_train = train_adata.X
    y_train = train_adata.obs['cell_type'].values
    
    # Prepare test data
    X_test = test_adata.X
    y_test = test_adata.obs['cell_type'].values
    
    # Check if this is cross-dataset analysis (different numbers of unique cell types)
    train_types = set(y_train)
    test_types = set(y_test)
    is_cross_dataset = len(train_types & test_types) < min(len(train_types), len(test_types)) * 0.5
    
    if is_cross_dataset:
        vprint(f"Cross-dataset detected: harmonizing cell type labels")
        vprint(f"Original - Train: {len(train_types)} types, Test: {len(test_types)} types")
        y_train, y_test = harmonize_cell_types(y_train, y_test)
        vprint(f"Harmonized - Train: {len(set(y_train))} types, Test: {len(set(y_test))} types")
        vprint(f"Common types: {len(set(y_train) & set(y_test))}")
    
    vprint(f"Training: {X_train.shape[0]} cells × {X_train.shape[1]} genes")
    vprint(f"Test: {X_test.shape[0]} cells × {X_test.shape[1]} genes")
    
    # Find common genes between datasets
    common_genes = train_adata.var_names.intersection(test_adata.var_names)
    vprint(f"Common genes: {len(common_genes)}")
    
    if len(common_genes) < 100:
        vprint("Warning: Very few common genes. Results may be unreliable.")
    
    # Subset to common genes
    train_subset = train_adata[:, common_genes].copy()
    test_subset = test_adata[:, common_genes].copy()
    
    X_train = train_subset.X
    X_test = test_subset.X
    
    vprint(f"Final dimensions: {X_train.shape[1]} common genes")
    
    # Encode labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    
    # Filter test set to common labels
    common_labels = np.isin(y_test, le.classes_)
    X_test = X_test[common_labels]
    y_test = y_test[common_labels]
    y_test_encoded = le.transform(y_test)
    
    vprint(f"Classes: {len(le.classes_)}")
    vprint(f"Final test set: {X_test.shape[0]} cells")
    
    # PCA
    vprint(f"Applying PCA ({n_components} components)...")
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    explained_var = pca.explained_variance_ratio_.sum()
    vprint(f"Explained variance: {explained_var:.3f}")
    
    # Define ML models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM (RBF)': SVC(kernel='rbf', random_state=42, probability=True),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
        'AdaBoost': AdaBoostClassifier(random_state=42, n_estimators=50),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
    }
    
    # Add Keras models if available
    keras_models = {}
    if KERAS_AVAILABLE:
        keras_models = {
            'Keras MLP': 'keras_mlp',
            'Keras 1D CNN': 'keras_cnn'
        }
    
    total_models = len(models) + len(keras_models)
    vprint(f"Training and evaluating {total_models} ML models...")
    
    # Train and evaluate models
    results = {}
    
    # Train traditional ML models
    for name, model in models.items():
        vprint(f"Training {name}...")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_pca, y_train_encoded, 
                                   cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                                   scoring='accuracy')
        
        # Train on full training set
        model.fit(X_train_pca, y_train_encoded)
        
        # Predictions
        train_pred = model.predict(X_train_pca)
        test_pred = model.predict(X_test_pca)
        
        # Calculate comprehensive metrics
        train_accuracy = accuracy_score(y_train_encoded, train_pred)
        test_accuracy = accuracy_score(y_test_encoded, test_pred)
        
        # Get prediction probabilities for ROC/PR curves
        if hasattr(model, "predict_proba"):
            test_proba = model.predict_proba(X_test_pca)
            train_proba = model.predict_proba(X_train_pca)
        elif hasattr(model, "decision_function"):
            # For SVM, convert decision function to probabilities
            from sklearn.calibration import CalibratedClassifierCV
            calibrated = CalibratedClassifierCV(model, cv=3)
            calibrated.fit(X_train_pca, y_train_encoded)
            test_proba = calibrated.predict_proba(X_test_pca)
            train_proba = calibrated.predict_proba(X_train_pca)
        else:
            test_proba = None
            train_proba = None
        
        # Calculate precision, recall, F1 for test set
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test_encoded, test_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            y_test_encoded, test_pred, average=None, zero_division=0)
        
        # Specificity calculation (manual)
        cm = confusion_matrix(y_test_encoded, test_pred)
        specificity_per_class = []
        for i in range(len(le.classes_)):
            if i < cm.shape[0] and i < cm.shape[1]:
                tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
                fp = cm[:, i].sum() - cm[i, i]
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                specificity_per_class.append(specificity)
            else:
                specificity_per_class.append(0)
        
        # Ensure specificity_per_class and weights have compatible shapes
        weights = np.bincount(y_test_encoded)
        if len(specificity_per_class) == len(weights):
            specificity_weighted = np.average(specificity_per_class, weights=weights)
        else:
            # Fallback to simple average if shapes don't match
            specificity_weighted = np.mean(specificity_per_class)
        
        # ROC-AUC and PR-AUC (for multiclass)
        if test_proba is not None and len(le.classes_) > 1:
            try:
                # One-vs-rest approach for multiclass
                if len(le.classes_) == 2:
                    # Binary case
                    y_test_binarized = label_binarize(y_test_encoded, classes=range(len(le.classes_)))
                    y_test_binarized = np.hstack([1 - y_test_binarized, y_test_binarized])
                else:
                    # Multiclass case
                    y_test_binarized = label_binarize(y_test_encoded, classes=range(len(le.classes_)))
                
                # Ensure probabilities match the binarized shape
                if test_proba.shape[1] == y_test_binarized.shape[1]:
                    roc_auc = roc_auc_score(y_test_binarized, test_proba, average='weighted', multi_class='ovr')
                    # Handle PR-AUC calculation more carefully
                    if len(np.unique(y_test_encoded)) > 1:  # Ensure multiple classes present
                        pr_auc = average_precision_score(y_test_binarized, test_proba, average='weighted')
                    else:
                        pr_auc = 0.0
                else:
                    vprint(f"Warning: Probability shape mismatch for {name}: {test_proba.shape[1]} vs {y_test_binarized.shape[1]}")
                    roc_auc = 0.0
                    pr_auc = 0.0
            except Exception as e:
                vprint(f"Warning: Could not calculate AUC metrics for {name}: {str(e)[:50]}...")
                roc_auc = 0.0
                pr_auc = 0.0
        else:
            roc_auc = 0.0
            pr_auc = 0.0
        
        results[name] = {
            'model': model,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity_weighted,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'specificity_per_class': specificity_per_class,
            'train_predictions': train_pred,
            'test_predictions': test_pred,
            'test_probabilities': test_proba,
            'train_probabilities': train_proba
        }
        
        vprint(f"CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        vprint(f"Train: {train_accuracy:.3f}")
        vprint(f"Test: {test_accuracy:.3f}")
        vprint(f"Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")
        vprint(f"Specificity: {specificity_weighted:.3f} | ROC-AUC: {roc_auc:.3f} | PR-AUC: {pr_auc:.3f}")
    
    # Train Keras models
    for name, model_type in keras_models.items():
        vprint(f"Training {name}...")
        
        # Create model
        if model_type == 'keras_mlp':
            keras_model = create_keras_mlp(X_train_pca.shape[1], len(le.classes_))
        elif model_type == 'keras_cnn':
            keras_model = create_keras_cnn(X_train_pca.shape[1], len(le.classes_))
        else:
            continue
            
        if keras_model is None:
            continue
        
        # Manual 5-fold cross-validation for Keras
        from sklearn.model_selection import KFold
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        
        for train_idx, val_idx in kfold.split(X_train_pca):
            X_tr, X_val = X_train_pca[train_idx], X_train_pca[val_idx]
            y_tr, y_val = y_train_encoded[train_idx], y_train_encoded[val_idx]
            
            # Create temporary model for CV
            if model_type == 'keras_mlp':
                temp_model = create_keras_mlp(X_train_pca.shape[1], len(le.classes_))
                epochs = 50
            else:  # keras_cnn
                temp_model = create_keras_cnn(X_train_pca.shape[1], len(le.classes_))
                epochs = 30
                
            temp_model.fit(X_tr, y_tr, epochs=epochs, batch_size=32, verbose=0)
            _, acc = temp_model.evaluate(X_val, y_val, verbose=0)
            cv_scores.append(acc)
        
        # Train final model on full training data
        if model_type == 'keras_mlp':
            epochs = 50
        else:  # keras_cnn
            epochs = 30
            
        keras_model.fit(X_train_pca, y_train_encoded, epochs=epochs, batch_size=32, verbose=0)
        
        # Get predictions
        train_pred_proba = keras_model.predict(X_train_pca, verbose=0)
        test_pred_proba = keras_model.predict(X_test_pca, verbose=0)
        
        train_pred = np.argmax(train_pred_proba, axis=1)
        test_pred = np.argmax(test_pred_proba, axis=1)
        
        # Calculate comprehensive metrics (same as traditional models)
        train_accuracy = accuracy_score(y_train_encoded, train_pred)
        test_accuracy = accuracy_score(y_test_encoded, test_pred)
        
        # Calculate precision, recall, F1 for test set
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test_encoded, test_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            y_test_encoded, test_pred, average=None, zero_division=0)
        
        # Specificity calculation
        cm = confusion_matrix(y_test_encoded, test_pred)
        specificity_per_class = []
        for i in range(len(le.classes_)):
            if i < cm.shape[0] and i < cm.shape[1]:
                tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
                fp = cm[:, i].sum() - cm[i, i]
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                specificity_per_class.append(specificity)
            else:
                specificity_per_class.append(0)
        
        # Ensure specificity_per_class and weights have compatible shapes
        weights = np.bincount(y_test_encoded)
        if len(specificity_per_class) == len(weights):
            specificity_weighted = np.average(specificity_per_class, weights=weights)
        else:
            # Fallback to simple average if shapes don't match
            specificity_weighted = np.mean(specificity_per_class)
        
        # ROC-AUC and PR-AUC using prediction probabilities
        try:
            if len(le.classes_) == 2:
                y_test_binarized = label_binarize(y_test_encoded, classes=range(len(le.classes_)))
                y_test_binarized = np.hstack([1 - y_test_binarized, y_test_binarized])
            else:
                y_test_binarized = label_binarize(y_test_encoded, classes=range(len(le.classes_)))
            
            if test_pred_proba.shape[1] == y_test_binarized.shape[1]:
                roc_auc = roc_auc_score(y_test_binarized, test_pred_proba, average='weighted', multi_class='ovr')
                # Handle PR-AUC calculation more carefully
                if len(np.unique(y_test_encoded)) > 1:  # Ensure multiple classes present
                    pr_auc = average_precision_score(y_test_binarized, test_pred_proba, average='weighted')
                else:
                    pr_auc = 0.0
            else:
                vprint(f"Warning: Probability shape mismatch for {name}: {test_pred_proba.shape[1]} vs {y_test_binarized.shape[1]}")
                roc_auc = 0.0
                pr_auc = 0.0
        except Exception as e:
            vprint(f"Warning: Could not calculate AUC metrics for {name}: {str(e)[:50]}...")
            roc_auc = 0.0
            pr_auc = 0.0
        
        # Store results
        results[name] = {
            'model': keras_model,
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity_weighted,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'specificity_per_class': specificity_per_class,
            'train_predictions': train_pred,
            'test_predictions': test_pred,
            'test_probabilities': test_pred_proba,
            'train_probabilities': train_pred_proba
        }
        
        vprint(f"CV: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
        vprint(f"Train: {train_accuracy:.3f}")
        vprint(f"Test: {test_accuracy:.3f}")
        vprint(f"Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")
        vprint(f"Specificity: {specificity_weighted:.3f} | ROC-AUC: {roc_auc:.3f} | PR-AUC: {pr_auc:.3f}")
    
    # Add k-NN Transfer Learning (Custom Implementation)
    if include_transfer_learning:
        vprint(f"Training k-NN Transfer Learning (Custom Implementation)...")
        
        # Use original RAW data for transfer learning (avoid double normalization)
        if train_adata_orig is not None and test_adata_orig is not None:
            # Use raw counts if available, otherwise use current data
            if hasattr(train_adata_orig, 'raw') and train_adata_orig.raw is not None:
                vprint(f"Using raw counts from adata.raw for reference")
                X_ref_orig = train_adata_orig.raw.X
                genes_ref_orig = train_adata_orig.raw.var_names.values
            else:
                vprint(f"Using current data for reference (no raw available)")
                X_ref_orig = train_adata_orig.X
                genes_ref_orig = train_adata_orig.var_names.values
                
            y_ref_orig = train_adata_orig.obs['cell_type'].values
            
            if hasattr(test_adata_orig, 'raw') and test_adata_orig.raw is not None:
                vprint(f"Using raw counts from adata.raw for query")
                X_query_orig = test_adata_orig.raw.X
                genes_query_orig = test_adata_orig.raw.var_names.values
            else:
                vprint(f"Using current data for query (no raw available)")
                X_query_orig = test_adata_orig.X
                genes_query_orig = test_adata_orig.var_names.values
                
            y_query_orig = test_adata_orig.obs['cell_type'].values
        else:
            # Fallback to processed data if original not available
            vprint(f"Using processed data as fallback")
            X_ref_orig = train_adata.X
            y_ref_orig = train_adata.obs['cell_type'].values
            genes_ref_orig = train_adata.var_names.values
            
            X_query_orig = test_adata.X  
            y_query_orig = test_adata.obs['cell_type'].values
            genes_query_orig = test_adata.var_names.values
        
        # Run transfer learning
        transfer_pred, confidence_scores = run_transfer_learning_knn(
            X_ref_orig, y_ref_orig, genes_ref_orig,
            X_query_orig, genes_query_orig, k=15
        )
        
        if transfer_pred is not None:
            try:
                # Convert transfer learning predictions to encoded labels that match test set
                # Only keep predictions for cells that have valid labels in test set
                valid_labels = np.isin(y_query_orig, le.classes_)
                y_query_filtered = y_query_orig[valid_labels]
                transfer_pred_filtered = transfer_pred[valid_labels]
                confidence_filtered = confidence_scores[valid_labels] if confidence_scores is not None else None
                
                # Encode transfer predictions to match our label encoder
                y_transfer_encoded = le.transform(transfer_pred_filtered)
                y_query_encoded = le.transform(y_query_filtered)
                
                # Calculate metrics
                transfer_accuracy = accuracy_score(y_query_encoded, y_transfer_encoded)
                
                # Calculate comprehensive metrics
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_query_encoded, y_transfer_encoded, average='weighted', zero_division=0)
                
                # Per-class metrics
                precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
                    y_query_encoded, y_transfer_encoded, average=None, zero_division=0)
                
                # Specificity calculation
                cm = confusion_matrix(y_query_encoded, y_transfer_encoded)
                specificity_per_class = []
                for i in range(len(le.classes_)):
                    if i < cm.shape[0] and i < cm.shape[1]:
                        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
                        fp = cm[:, i].sum() - cm[i, i]
                        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                        specificity_per_class.append(specificity)
                    else:
                        specificity_per_class.append(0)
                
                # Ensure specificity_per_class and weights have compatible shapes
                weights = np.bincount(y_query_encoded)
                if len(specificity_per_class) == len(weights):
                    specificity_weighted = np.average(specificity_per_class, weights=weights)
                else:
                    # Fallback to simple average if shapes don't match
                    specificity_weighted = np.mean(specificity_per_class)
                
                # For transfer learning, we don't have traditional CV or probability scores
                # But we can use confidence scores as a proxy
                cv_mean = transfer_accuracy  # Use accuracy as CV proxy
                cv_std = 0.0  # No CV for transfer learning
                
                # Store results 
                results['k-NN Transfer Learning'] = {
                    'model': 'transfer_learning',
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'train_accuracy': transfer_accuracy,  # Same as test for transfer learning
                    'test_accuracy': transfer_accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'specificity': specificity_weighted,
                    'roc_auc': 0.0,  # Not applicable for transfer learning
                    'pr_auc': 0.0,   # Not applicable for transfer learning
                    'precision_per_class': precision_per_class,
                    'recall_per_class': recall_per_class,
                    'f1_per_class': f1_per_class,
                    'specificity_per_class': specificity_per_class,
                    'train_predictions': y_transfer_encoded,  # Same as test for transfer learning
                    'test_predictions': y_transfer_encoded,
                    'test_probabilities': None,  # Transfer learning doesn't provide probabilities
                    'train_probabilities': None,
                    'confidence_scores': confidence_filtered
                }
                
                vprint(f"Accuracy: {transfer_accuracy:.3f}")
                vprint(f"Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")
                vprint(f"Specificity: {specificity_weighted:.3f}")
                if confidence_filtered is not None:
                    vprint(f"Avg Confidence: {np.mean(confidence_filtered):.3f}")
                vprint(f"🏆 GOLD STANDARD for cross-dataset validation")
                
            except Exception as e:
                vprint(f"Transfer learning label processing failed: {e}")
        else:
            vprint(f"Transfer learning could not be performed")
    
    # Add Scanpy Ingest Transfer Learning (Standard Method)
    if include_transfer_learning and train_adata_orig is not None and test_adata_orig is not None:
        vprint(f"Training Scanpy Ingest Transfer Learning (Standard Method)...")
        
        # Use copies of original datasets to preserve data integrity
        train_adata_copy = train_adata_orig.copy()
        test_adata_copy = test_adata_orig.copy()
        
        # Apply harmonization if detected as cross-dataset
        if is_cross_dataset:
            vprint(f"Applying label harmonization for ingest...")
            train_labels_harmonized, test_labels_harmonized = harmonize_cell_types(
                train_adata_copy.obs['cell_type'].values,
                test_adata_copy.obs['cell_type'].values
            )
            train_adata_copy.obs['cell_type'] = train_labels_harmonized
            test_adata_copy.obs['cell_type'] = test_labels_harmonized
        
        # Apply scanpy ingest
        predicted_labels_ingest, confidence_ingest = scanpy_ingest_transfer_learning(
            train_adata_copy, test_adata_copy, use_rep='X_pca', k=50
        )
        
        if predicted_labels_ingest is not None:
            try:
                # Encode labels
                le_ingest = LabelEncoder()
                y_train_ingest = le_ingest.fit_transform(train_adata_copy.obs['cell_type'].values)
                
                # Handle unseen labels in test set
                test_labels_filtered = []
                test_indices_valid = []
                for i, label in enumerate(predicted_labels_ingest):
                    if label in le_ingest.classes_:
                        test_labels_filtered.append(label)
                        test_indices_valid.append(i)
                
                if len(test_labels_filtered) > 0:
                    y_ingest_encoded = le_ingest.transform(test_labels_filtered)
                    confidence_ingest_filtered = confidence_ingest[test_indices_valid] if confidence_ingest is not None else None
                    
                    # Get true labels for comparison
                    y_test_true_ingest = test_adata_copy.obs['cell_type'].values
                    y_test_true_filtered = [y_test_true_ingest[i] for i in test_indices_valid]
                    y_test_true_filtered = [label for label in y_test_true_filtered if label in le_ingest.classes_]
                    
                    if len(y_test_true_filtered) > 0:
                        y_test_encoded_ingest = le_ingest.transform(y_test_true_filtered)
                        
                        # Calculate metrics
                        ingest_accuracy = accuracy_score(y_test_encoded_ingest, y_ingest_encoded)
                        precision, recall, f1, _ = precision_recall_fscore_support(
                            y_test_encoded_ingest, y_ingest_encoded, average='weighted', zero_division=0)
                        
                        # Per-class metrics
                        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
                            y_test_encoded_ingest, y_ingest_encoded, average=None, zero_division=0)
                        
                        # Specificity calculation
                        cm = confusion_matrix(y_test_encoded_ingest, y_ingest_encoded)
                        specificity_per_class = []
                        for i in range(len(le_ingest.classes_)):
                            if i < cm.shape[0] and i < cm.shape[1]:
                                tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
                                fp = np.sum(cm[:, i]) - cm[i, i]
                                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                                specificity_per_class.append(specificity)
                            else:
                                specificity_per_class.append(0.0)
                        
                        # Ensure specificity_per_class and weights have compatible shapes
                        weights = np.bincount(y_test_encoded_ingest)
                        if len(specificity_per_class) == len(weights):
                            specificity_weighted = np.average(specificity_per_class, weights=weights)
                        else:
                            # Fallback to simple average if shapes don't match
                            specificity_weighted = np.mean(specificity_per_class)
                        
                        # Store results
                        results['Scanpy Ingest Transfer Learning (Standard Method)'] = {
                            'model': 'scanpy_ingest',
                            'cv_mean': ingest_accuracy,  # Use accuracy as CV proxy
                            'cv_std': 0.0,  # No CV for transfer learning
                            'train_accuracy': ingest_accuracy,  # Same as test for transfer learning
                            'test_accuracy': ingest_accuracy,
                            'precision': precision,
                            'recall': recall,
                            'f1_score': f1,
                            'specificity': specificity_weighted,
                            'roc_auc': 0.0,  # Not applicable for transfer learning
                            'pr_auc': 0.0,   # Not applicable for transfer learning
                            'precision_per_class': precision_per_class,
                            'recall_per_class': recall_per_class,
                            'f1_per_class': f1_per_class,
                            'specificity_per_class': specificity_per_class,
                            'train_predictions': y_ingest_encoded,  # Same as test for transfer learning
                            'test_predictions': y_ingest_encoded,
                            'test_probabilities': None,  # Transfer learning doesn't provide probabilities
                            'train_probabilities': None,
                            'confidence_scores': confidence_ingest_filtered
                        }
                        
                        vprint(f"Accuracy: {ingest_accuracy:.3f}")
                        vprint(f"Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")
                        vprint(f"Specificity: {specificity_weighted:.3f}")
                        if confidence_ingest_filtered is not None:
                            vprint(f"Avg Confidence: {np.mean(confidence_ingest_filtered):.3f}")
                        vprint(f"📊 STANDARD METHOD: Scanpy sc.tl.ingest")
                        
                    else:
                        vprint(f"No valid test labels after filtering")
                else:
                    vprint(f"No valid predictions after label filtering")
                    
            except Exception as e:
                vprint(f"Scanpy ingest label processing failed: {e}")
        else:
            vprint(f"Scanpy ingest could not be performed")
    
    # Create visualizations
    create_ml_visualizations(results, le, y_test_encoded, output_dir)
    create_enhanced_metrics_plots(results, le, y_test_encoded, output_dir)
    
    # Save results
    save_ml_results(results, le, output_dir)
    
    return results, le

def create_ml_visualizations(results, le, y_test_encoded, output_dir):
    """Create comprehensive ML visualization plots"""
    vprint("Creating ML visualizations...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data
    model_names = list(results.keys())
    cv_means = [results[name]['cv_mean'] for name in model_names]
    cv_stds = [results[name]['cv_std'] for name in model_names]
    train_accs = [results[name]['train_accuracy'] for name in model_names]
    test_accs = [results[name]['test_accuracy'] for name in model_names]
    
    # Additional metrics
    precisions = [results[name]['precision'] for name in model_names]
    recalls = [results[name]['recall'] for name in model_names]
    f1_scores = [results[name]['f1_score'] for name in model_names]
    specificities = [results[name]['specificity'] for name in model_names]
    roc_aucs = [results[name]['roc_auc'] for name in model_names]
    pr_aucs = [results[name]['pr_auc'] for name in model_names]
    
    # Sort by test accuracy
    sorted_indices = np.argsort(test_accs)[::-1]
    
    # Figure 1: Performance comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
    
    # Test accuracy ranking
    sorted_names = [model_names[i] for i in sorted_indices]
    sorted_test_accs = [test_accs[i] for i in sorted_indices]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
    bars1 = ax1.barh(range(len(sorted_names)), sorted_test_accs, color=colors, alpha=0.8)
    ax1.set_yticks(range(len(sorted_names)))
    ax1.set_yticklabels(sorted_names)
    ax1.set_xlabel('Test Accuracy')
    ax1.set_title('Model Performance Ranking', fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Add accuracy values
    for i, (bar, acc) in enumerate(zip(bars1, sorted_test_accs)):
        ax1.text(acc + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{acc:.3f}', va='center', fontweight='bold')
    
    # CV vs Test performance
    ax2.scatter(cv_means, test_accs, s=100, alpha=0.7, color='coral')
    ax2.plot([min(cv_means), max(cv_means)], [min(cv_means), max(cv_means)], 
             'k--', alpha=0.5, label='Perfect correlation')
    ax2.set_xlabel('Cross-Validation Accuracy')
    ax2.set_ylabel('Test Accuracy')
    ax2.set_title('CV vs Test Performance', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Performance with error bars
    x_pos = range(len(model_names))
    ax3.errorbar(x_pos, cv_means, yerr=cv_stds, fmt='o', capsize=5, 
                color='green', alpha=0.7, label='CV Score', markersize=8)
    ax3.scatter(x_pos, test_accs, color='red', alpha=0.7, label='Test Accuracy', s=80)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(model_names, rotation=45, ha='right')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Cross-Validation vs Test Performance', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Train vs Test accuracy (overfitting check)
    ax4.scatter(train_accs, test_accs, s=100, alpha=0.7, color='purple')
    ax4.plot([min(train_accs), max(train_accs)], [min(train_accs), max(train_accs)], 
             'k--', alpha=0.5, label='No overfitting line')
    ax4.set_xlabel('Training Accuracy')
    ax4.set_ylabel('Test Accuracy')
    ax4.set_title('Training vs Test Accuracy', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/performance_comparison.pdf', bbox_inches='tight')
    plt.close()
    
    # Figure 2: Confusion matrices for top 3 models
    sorted_results = sorted(results.items(), key=lambda x: x[1]['test_accuracy'], reverse=True)
    top_3_models = sorted_results[:3]
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Get unique classes in test set
    unique_test_classes = np.unique(y_test_encoded)
    test_class_names = le.classes_[unique_test_classes]
    
    for i, (name, result) in enumerate(top_3_models):
        cm = confusion_matrix(y_test_encoded, result['test_predictions'], labels=unique_test_classes)
        
        # Normalize confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        im = axes[i].imshow(cm_norm, interpolation='nearest', cmap='Blues')
        axes[i].set_title(f'{name}Accuracy: {result["test_accuracy"]:.3f}', fontweight='bold')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        
        # Add text annotations
        thresh = cm_norm.max() / 2.
        for j in range(cm.shape[0]):
            for k in range(cm.shape[1]):
                axes[i].text(k, j, f'{cm[j, k]}',
                           ha="center", va="center",
                           color="white"if cm_norm[j, k] > thresh else "black",
                           fontsize=8)
        
        axes[i].set_ylabel('True Label')
        axes[i].set_xlabel('Predicted Label')
        axes[i].set_xticks(range(len(test_class_names)))
        axes[i].set_yticks(range(len(test_class_names)))
        axes[i].set_xticklabels(test_class_names, rotation=45, ha='right')
        axes[i].set_yticklabels(test_class_names)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrices_top3.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/confusion_matrices_top3.pdf', bbox_inches='tight')
    plt.close()
    
    vprint(f"ML visualizations saved to: {output_dir}")

def create_enhanced_metrics_plots(results, le, y_test_encoded, output_dir):
    """Create comprehensive metrics visualization plots"""
    vprint("Creating enhanced metrics visualizations...")
    
    # Prepare data
    model_names = list(results.keys())
    test_accs = [results[name]['test_accuracy'] for name in model_names]
    precisions = [results[name]['precision'] for name in model_names]
    recalls = [results[name]['recall'] for name in model_names]
    f1_scores = [results[name]['f1_score'] for name in model_names]
    specificities = [results[name]['specificity'] for name in model_names]
    roc_aucs = [results[name]['roc_auc'] for name in model_names]
    pr_aucs = [results[name]['pr_auc'] for name in model_names]
    
    # Sort by F1 score for better ranking
    sorted_indices = np.argsort(f1_scores)[::-1]
    
    # Figure 1: Comprehensive Performance Metrics
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Multi-metric ranking
    sorted_names = [model_names[i] for i in sorted_indices]
    metrics_to_plot = {
        'Accuracy': [test_accs[i] for i in sorted_indices],
        'Precision': [precisions[i] for i in sorted_indices],
        'Recall': [recalls[i] for i in sorted_indices],
        'F1-Score': [f1_scores[i] for i in sorted_indices],
        'Specificity': [specificities[i] for i in sorted_indices]
    }
    
    x = np.arange(len(sorted_names))
    width = 0.15
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (metric_name, values) in enumerate(metrics_to_plot.items()):
        bars = ax1.bar(x + i*width, values, width, label=metric_name, 
                      color=colors[i], alpha=0.8)
        # Add value labels on bars
        for bar, val in zip(bars, values):
            if val > 0.05:  # Only label if bar is tall enough
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=8, rotation=0)
    
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Score')
    ax1.set_title('Comprehensive Performance Metrics Comparison', fontweight='bold', fontsize=14)
    ax1.set_xticks(x + width*2)
    ax1.set_xticklabels(sorted_names, rotation=45, ha='right')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.set_ylim(0, 1.1)
    ax1.grid(axis='y', alpha=0.3)
    
    # AUC metrics comparison
    sorted_roc_aucs = [roc_aucs[i] for i in sorted_indices]
    sorted_pr_aucs = [pr_aucs[i] for i in sorted_indices]
    
    x_auc = np.arange(len(sorted_names))
    width_auc = 0.35
    
    bars1 = ax2.bar(x_auc - width_auc/2, sorted_roc_aucs, width_auc, 
                   label='ROC-AUC', color='skyblue', alpha=0.8)
    bars2 = ax2.bar(x_auc + width_auc/2, sorted_pr_aucs, width_auc,
                   label='PR-AUC', color='lightcoral', alpha=0.8)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.05:
                ax2.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax2.set_xlabel('Models')
    ax2.set_ylabel('AUC Score')
    ax2.set_title('ROC-AUC vs PR-AUC Comparison', fontweight='bold', fontsize=14)
    ax2.set_xticks(x_auc)
    ax2.set_xticklabels(sorted_names, rotation=45, ha='right')
    ax2.legend()
    ax2.set_ylim(0, 1.1)
    ax2.grid(axis='y', alpha=0.3)
    
    # Precision-Recall scatter plot
    scatter = ax3.scatter(recalls, precisions, s=120, alpha=0.7, c=f1_scores, 
                         cmap='viridis', edgecolors='black', linewidths=1)
    
    for i, name in enumerate(model_names):
        ax3.annotate(name, (recalls[i], precisions[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    ax3.set_xlabel('Recall (Sensitivity)')
    ax3.set_ylabel('Precision')
    ax3.set_title('Precision-Recall Trade-off (colored by F1-score)', fontweight='bold', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-0.05, 1.05)
    ax3.set_ylim(-0.05, 1.05)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('F1-Score', rotation=270, labelpad=15)
    
    # ROC curve comparison for top 3 models
    ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
    
    # Get top 3 models by F1 score
    top_3_indices = sorted_indices[:3]
    colors_roc = ['red', 'blue', 'green']
    
    for i, idx in enumerate(top_3_indices):
        model_name = model_names[idx]
        result = results[model_name]
        
        if result['test_probabilities'] is not None:
            try:
                # For multiclass ROC, we'll plot macro-average
                if len(le.classes_) == 2:
                    y_test_binarized = label_binarize(y_test_encoded, classes=range(len(le.classes_)))
                    y_test_binarized = np.hstack([1 - y_test_binarized, y_test_binarized])
                else:
                    y_test_binarized = label_binarize(y_test_encoded, classes=range(len(le.classes_)))
                
                test_proba = result['test_probabilities']
                if test_proba.shape[1] == y_test_binarized.shape[1]:
                    # Calculate macro-average ROC
                    all_fpr = np.unique(np.concatenate([roc_curve(y_test_binarized[:, j], 
                                                                 test_proba[:, j])[0] 
                                                       for j in range(y_test_binarized.shape[1])]))
                    mean_tpr = np.zeros_like(all_fpr)
                    
                    for j in range(y_test_binarized.shape[1]):
                        fpr, tpr, _ = roc_curve(y_test_binarized[:, j], test_proba[:, j])
                        mean_tpr += np.interp(all_fpr, fpr, tpr)
                    
                    mean_tpr /= y_test_binarized.shape[1]
                    auc_score = result['roc_auc']
                    ax4.plot(all_fpr, mean_tpr, color=colors_roc[i], linewidth=2,
                            label=f'{model_name} (AUC = {auc_score:.2f})')
            except Exception as e:
                vprint(f"Warning: Could not plot ROC for {model_name}: {str(e)[:30]}...")
    
    ax4.set_xlabel('False Positive Rate')
    ax4.set_ylabel('True Positive Rate')
    ax4.set_title('ROC Curves - Top 3 Models (Macro-Average)', fontweight='bold', fontsize=14)
    ax4.legend(loc='lower right')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(-0.05, 1.05)
    ax4.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/enhanced_metrics.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/enhanced_metrics.pdf', bbox_inches='tight')
    plt.close()
    
    # Figure 2: Per-class metrics heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Get per-class data for top 5 models
    top_5_indices = sorted_indices[:5]
    top_5_names = [model_names[i] for i in top_5_indices]
    
    # Precision heatmap
    precision_matrix = []
    recall_matrix = []
    f1_matrix = []
    specificity_matrix = []
    
    for idx in top_5_indices:
        model_name = model_names[idx]
        result = results[model_name]
        
        # Ensure arrays have correct length
        n_classes = len(le.classes_)
        precision_per_class = result['precision_per_class']
        recall_per_class = result['recall_per_class']
        f1_per_class = result['f1_per_class']
        specificity_per_class = result['specificity_per_class']
        
        # Pad or truncate to match number of classes
        if len(precision_per_class) < n_classes:
            precision_per_class = np.pad(precision_per_class, (0, n_classes - len(precision_per_class)))
            recall_per_class = np.pad(recall_per_class, (0, n_classes - len(recall_per_class)))
            f1_per_class = np.pad(f1_per_class, (0, n_classes - len(f1_per_class)))
            specificity_per_class = specificity_per_class + [0] * (n_classes - len(specificity_per_class))
        elif len(precision_per_class) > n_classes:
            precision_per_class = precision_per_class[:n_classes]
            recall_per_class = recall_per_class[:n_classes]
            f1_per_class = f1_per_class[:n_classes]
            specificity_per_class = specificity_per_class[:n_classes]
        
        precision_matrix.append(precision_per_class)
        recall_matrix.append(recall_per_class)
        f1_matrix.append(f1_per_class)
        specificity_matrix.append(specificity_per_class)
    
    # Create combined metrics matrix
    combined_matrix = np.array(f1_matrix)
    
    im1 = ax1.imshow(combined_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
    ax1.set_xticks(range(len(le.classes_)))
    ax1.set_yticks(range(len(top_5_names)))
    ax1.set_xticklabels(le.classes_, rotation=45, ha='right')
    ax1.set_yticklabels(top_5_names)
    ax1.set_title('Per-Class F1-Scores Heatmap', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Cell Types')
    ax1.set_ylabel('Models')
    
    # Add text annotations
    for i in range(len(top_5_names)):
        for j in range(len(le.classes_)):
            text = ax1.text(j, i, f'{combined_matrix[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=10)
    
    plt.colorbar(im1, ax=ax1, label='F1-Score')
    
    # Specificity heatmap
    specificity_matrix = np.array(specificity_matrix)
    im2 = ax2.imshow(specificity_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
    ax2.set_xticks(range(len(le.classes_)))
    ax2.set_yticks(range(len(top_5_names)))
    ax2.set_xticklabels(le.classes_, rotation=45, ha='right')
    ax2.set_yticklabels(top_5_names)
    ax2.set_title('Per-Class Specificity Heatmap', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Cell Types')
    ax2.set_ylabel('Models')
    
    # Add text annotations
    for i in range(len(top_5_names)):
        for j in range(len(le.classes_)):
            text = ax2.text(j, i, f'{specificity_matrix[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=10)
    
    plt.colorbar(im2, ax=ax2, label='Specificity')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/per_class_metrics.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/per_class_metrics.pdf', bbox_inches='tight')
    plt.close()
    
    vprint(f"Enhanced metrics visualizations saved to: {output_dir}")

def save_ml_results(results, le, output_dir):
    """Save ML results to files"""
    vprint("Saving ML results...")
    
    # Model performance summary with comprehensive metrics
    performance_data = {
        'Model': list(results.keys()),
        'CV_Mean': [r['cv_mean'] for r in results.values()],
        'CV_Std': [r['cv_std'] for r in results.values()],
        'Train_Accuracy': [r['train_accuracy'] for r in results.values()],
        'Test_Accuracy': [r['test_accuracy'] for r in results.values()],
        'Precision': [r['precision'] for r in results.values()],
        'Recall': [r['recall'] for r in results.values()],
        'F1_Score': [r['f1_score'] for r in results.values()],
        'Specificity': [r['specificity'] for r in results.values()],
        'ROC_AUC': [r['roc_auc'] for r in results.values()],
        'PR_AUC': [r['pr_auc'] for r in results.values()]
    }
    
    performance_df = pd.DataFrame(performance_data)
    performance_df = performance_df.sort_values('F1_Score', ascending=False)
    performance_df.to_csv(f'{output_dir}/model_performance_summary.csv', index=False)
    
    # Also save detailed per-class metrics for top 3 models
    top_3_models = performance_df.head(3)['Model'].values
    per_class_data = []
    
    for model_name in top_3_models:
        result = results[model_name]
        for i, class_name in enumerate(le.classes_):
            if i < len(result['precision_per_class']):
                per_class_data.append({
                    'Model': model_name,
                    'Cell_Type': class_name,
                    'Precision': result['precision_per_class'][i],
                    'Recall': result['recall_per_class'][i],
                    'F1_Score': result['f1_per_class'][i],
                    'Specificity': result['specificity_per_class'][i] if i < len(result['specificity_per_class']) else 0
                })
    
    if per_class_data:
        per_class_df = pd.DataFrame(per_class_data)
        per_class_df.to_csv(f'{output_dir}/per_class_metrics.csv', index=False)
    
    # Cell type information
    cell_types_df = pd.DataFrame({'Cell_Type': le.classes_})
    cell_types_df.to_csv(f'{output_dir}/cell_types.csv', index=False)
    
    vprint(f"Results saved to: {output_dir}")

def run_same_dataset_analysis(approach):
    """Run same-dataset validation using Union HVGs for consistency with cross-dataset analysis"""
    vprint(f"Same-Dataset Analysis ({approach}) - Union HVG")
    vprint("="* 60)
    
    # Compute Union HVGs for consistency with cross-dataset analysis
    union_hvgs, available_datasets = compute_union_hvgs(approach)
    
    if union_hvgs is None or len(union_hvgs) < 100:
        vprint("Insufficient Union HVGs, falling back to standard approach")
        # Fallback to original approach
        datasets = load_processed_data(approach)
        if len(datasets) == 0:
            vprint("No datasets found for same-dataset analysis")
            return False
        dataset_name = list(datasets.keys())[0]
        adata = datasets[dataset_name]
        vprint(f"Using {dataset_name} for same-dataset analysis (standard HVGs)")
        adata_processed = preprocess_for_ml(adata.copy())
    else:
        # Load full processed dataset filtered to Union HVGs
        processed_data_path = f'data/{approach}'
        dataset_name = 'pbmc3k' if 'pbmc3k' in available_datasets else list(available_datasets)[0]
        
        file_path = f'{processed_data_path}/{dataset_name}_full_processed.h5ad'
        try:
            adata = sc.read(file_path)
            # Filter to Union HVGs only
            common_genes = [g for g in union_hvgs if g in adata.var_names]
            adata_filtered = adata[:, common_genes].copy()
            vprint(f"Using {dataset_name} for same-dataset analysis ({len(union_hvgs):,} Union HVGs)")
            vprint(f"Dataset: {adata_filtered.n_obs} cells × {adata_filtered.n_vars} genes")
            adata_processed = preprocess_for_ml_union_hvg(adata_filtered.copy())
        except Exception as e:
            vprint(f"Failed to load {dataset_name}: {e}")
            return False
    
    # Split into train/test
    train_idx, test_idx = train_test_split(range(adata_processed.n_obs), 
                                          test_size=0.3, random_state=42,
                                          stratify=adata_processed.obs['cell_type'])
    
    train_adata = adata_processed[train_idx].copy()
    test_adata = adata_processed[test_idx].copy()
    
    vprint(f"Train split: {train_adata.n_obs} cells × {train_adata.n_vars} genes")
    vprint(f"Test split: {test_adata.n_obs} cells × {test_adata.n_vars} genes")
    
    # Run ML analysis
    output_dir = f'figures/machine_learning_results/{approach}/same_dataset_union_hvg'
    results, le = run_ml_analysis(train_adata, test_adata, output_dir)
    
    return results

def compute_union_hvgs(approach):
    """Compute union of highly variable genes across all datasets"""
    vprint("Computing Union HVGs for fair cross-dataset comparison...")
    
    # Load processed datasets (HVG versions)
    hvg_datasets = {}
    processed_data_path = f'data/{approach}'
    
    # Find all processed (HVG) datasets using consistent filtering logic
    import glob
    all_processed_files = glob.glob(f'{processed_data_path}/*_processed.h5ad')
    processed_file_names = [f.split('/')[-1] for f in all_processed_files]
    
    # Use same filtering logic as load_processed_data
    filtered_files = []
    if approach == 'annotated':
        # Use consistent HVG versions (2000 genes each)
        priority_files = ['pbmc3k_processed.h5ad', 'pbmc_multiome_full_processed.h5ad']
        for pfile in priority_files:
            if pfile in processed_file_names:
                filtered_files.append(f'{processed_data_path}/{pfile}')
    else:
        # For not_annotated, exclude _full variants to avoid duplicates
        hvg_files = [f for f in all_processed_files if 'full_processed' not in f]
        filtered_files = hvg_files
    
    if not filtered_files:
        vprint(f"No suitable processed files found, using all available")
        filtered_files = all_processed_files
    
    for file_path in filtered_files:
        dataset_name = file_path.split('/')[-1].replace('_processed.h5ad', '')
        try:
            adata = sc.read(file_path)
            hvg_datasets[dataset_name] = set(adata.var_names)
            vprint(f"{dataset_name}: {len(adata.var_names):,} HVGs")
        except Exception as e:
            vprint(f"Failed to load {dataset_name}: {e}")
    
    if len(hvg_datasets) < 2:
        vprint("Need at least 2 datasets for Union HVG")
        return None, {}
    
    # Compute union of all HVGs
    union_hvgs = set()
    for dataset_name, hvg_set in hvg_datasets.items():
        union_hvgs = union_hvgs.union(hvg_set)
    
    vprint(f"Union HVGs: {len(union_hvgs):,} genes")
    
    # Load full datasets to check availability
    full_datasets = {}
    full_files = glob.glob(f'{processed_data_path}/*_full_processed.h5ad')
    
    for file_path in full_files:
        dataset_name = file_path.split('/')[-1].replace('_full_processed.h5ad', '')
        try:
            adata = sc.read(file_path)
            full_datasets[dataset_name] = set(adata.var_names)
        except Exception as e:
            vprint(f"Failed to load full {dataset_name}: {e}")
    
    # Find intersection of union HVGs with all full datasets
    available_union_hvgs = union_hvgs
    for dataset_name, gene_set in full_datasets.items():
        available_union_hvgs = available_union_hvgs.intersection(gene_set)
        vprint(f"{dataset_name} contains {len(union_hvgs.intersection(gene_set)):,}/{len(union_hvgs):,} union HVGs")
    
    vprint(f" Available in ALL datasets: {len(available_union_hvgs):,} genes")
    vprint(f"Union HVGs: {len(available_union_hvgs)}")
    
    return list(available_union_hvgs), full_datasets.keys()

def run_cross_dataset_analysis(approach):
    """Run cross-dataset validation: Train once on PBMC 3k (70/30 + CV), test on external datasets"""
    vprint(f"Cross-Dataset Analysis ({approach}) - Single Training with Union HVG")
    vprint("="* 70)
    
    # Compute Union HVGs for fair comparison across all datasets
    union_hvgs, available_datasets = compute_union_hvgs(approach)
    
    if union_hvgs is None or len(union_hvgs) < 100:
        vprint("Insufficient Union HVGs, falling back to standard approach")
        return run_cross_dataset_analysis_fallback(approach)
    
    # Load full processed datasets filtered to Union HVGs
    datasets = {}
    processed_data_path = f'data/{approach}'
    
    for dataset_name in available_datasets:
        file_path = f'{processed_data_path}/{dataset_name}_full_processed.h5ad'
        try:
            adata = sc.read(file_path)
            # Filter to Union HVGs only
            common_genes = [g for g in union_hvgs if g in adata.var_names]
            adata_filtered = adata[:, common_genes].copy()
            datasets[dataset_name] = adata_filtered
            vprint(f"Loaded {dataset_name}: {adata_filtered.n_obs} cells × {adata_filtered.n_vars} genes")
        except Exception as e:
            vprint(f"Failed to load {dataset_name}: {e}")
    
    if len(datasets) < 2:
        vprint("Need at least 2 datasets for cross-dataset analysis")
        return False
    
    dataset_names = list(datasets.keys())
    
    # Use PBMC 3k as training dataset (industry standard reference)
    preferred_reference = 'pbmc3k'
    if preferred_reference in dataset_names:
        train_name = preferred_reference
        test_names = [name for name in dataset_names if name != preferred_reference]
    else:
        # Fallback to first dataset if PBMC 3k not available
        train_name = dataset_names[0]
        test_names = dataset_names[1:]
    
    vprint(f"🎯 TRAINING PHASE")
    vprint(f"Dataset: {train_name} ({len(union_hvgs):,} Union HVGs)")
    
    # === SINGLE TRAINING PHASE ===
    # Get training dataset and split it 70/30
    train_adata_full = datasets[train_name].copy()
    train_adata_processed = preprocess_for_ml_union_hvg(train_adata_full.copy())
    
    # 70/30 split for proper ML validation
    train_idx, val_idx = train_test_split(range(train_adata_processed.n_obs), 
                                         test_size=0.3, random_state=42,
                                         stratify=train_adata_processed.obs['cell_type'])
    
    train_split = train_adata_processed[train_idx].copy()
    val_split = train_adata_processed[val_idx].copy()
    
    vprint(f"Train split: {train_split.n_obs} cells × {train_split.n_vars} genes")
    vprint(f"Validation split: {val_split.n_obs} cells × {val_split.n_vars} genes")
    
    # Train all models on PBMC 3k training split with cross-validation
    output_dir_training = f'figures/machine_learning_results/{approach}/cross_dataset_training_validation'
    training_results, le = run_ml_analysis(train_split, val_split, output_dir_training, 
                                         include_transfer_learning=True,
                                         train_adata_orig=train_adata_full[train_idx], 
                                         test_adata_orig=train_adata_full[val_idx])
    
    vprint(f" Training completed on {train_name}")
    
    # === EXTERNAL DATASET TESTING PHASE ===
    vprint(f"EXTERNAL TESTING PHASE")
    vprint("Applying trained models to external datasets...")
    
    all_results = {'training_validation': training_results}
    
    for test_name in test_names:
        vprint(f"→ Testing on: {test_name}")
        
        test_adata = datasets[test_name].copy()
        test_adata_processed = preprocess_for_ml_union_hvg(test_adata.copy())
        
        vprint(f"External test data: {test_adata_processed.n_obs} cells × {test_adata_processed.n_vars} genes")
        
        # Apply the SAME trained models to external dataset
        # Note: We use the full training dataset (not just 70%) for external testing
        train_adata_full_processed = preprocess_for_ml_union_hvg(train_adata_full.copy())
        
        output_dir = f'figures/machine_learning_results/{approach}/cross_dataset_{test_name}_union_hvg'
        
        # Run analysis: train on full PBMC 3k, test on external dataset
        results, _ = run_ml_analysis(train_adata_full_processed, test_adata_processed, output_dir, 
                                   include_transfer_learning=True,
                                   train_adata_orig=train_adata_full, 
                                   test_adata_orig=test_adata)
        
        all_results[test_name] = results
        vprint(f"External testing completed on {test_name}")
    
    return all_results

def preprocess_for_ml_union_hvg(adata, n_top_genes=2000):
    """Preprocess data for ML when using Union HVGs (skip HVG selection)"""
    vprint(f"Preprocessing {adata.n_obs} cells × {adata.n_vars} genes (Union HVG)...")
    
    # Remove cells with missing cell types
    adata = adata[~adata.obs['cell_type'].isna()].copy()
    adata = adata[adata.obs['cell_type'] != 'filtered'].copy()
    
    # Remove cells with Unknown cell type annotation  
    original_cells = adata.n_obs
    adata = adata[adata.obs['cell_type'] != 'Unknown'].copy()
    unknown_filtered = original_cells - adata.n_obs
    if unknown_filtered > 0:
        vprint(f"Filtered out {unknown_filtered} Unknown cells ({unknown_filtered/original_cells*100:.1f}%)")
    
    vprint(f"After filtering: {adata.n_obs} cells × {adata.n_vars} genes")
    
    # Check if data is already normalized (it should be from EDA processing)
    if hasattr(adata, 'raw') and adata.raw is not None:
        vprint("Data already processed in EDA - using existing normalization")
        # Data is already normalized and log-transformed from EDA
        # Skip HVG selection since we're using Union HVGs
        vprint(f"Using Union HVGs: {adata.n_vars} genes")
        
        # Scale data
        sc.pp.scale(adata, max_value=10)
    else:
        vprint("Applying fresh normalization (Union HVG approach)")
        # Store raw data
        adata.raw = adata
        
        # Normalize and log transform
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        
        # Skip HVG selection - we're already using Union HVGs
        vprint(f"Using Union HVGs: {adata.n_vars} genes")
        
        # Scale data
        sc.pp.scale(adata, max_value=10)
    
    return adata

def run_cross_dataset_analysis_fallback(approach):
    """Fallback to original cross-dataset analysis if Union HVG fails"""
    vprint("Using fallback cross-dataset analysis...")
    
    # Special handling for annotated approach with gene count mismatch
    if approach == 'annotated':
        vprint("Annotated approach has inconsistent gene counts between datasets")
        vprint("Cross-dataset analysis not reliable with current data")
        vprint("Recommendation: Re-run EDA with consistent HVG selection across datasets")
        return {"status": "skipped", "reason": "inconsistent_gene_counts"}
    
    # Load processed data
    datasets = load_processed_data(approach)
    
    if len(datasets) < 2:
        vprint("Need at least 2 datasets for cross-dataset analysis")
        return False
    
    dataset_names = list(datasets.keys())
    
    # Use PBMC 3k as reference/training dataset
    preferred_reference = 'pbmc3k'
    if preferred_reference in dataset_names:
        train_name = preferred_reference
        test_names = [name for name in dataset_names if name != preferred_reference]
    else:
        train_name = dataset_names[0]
        test_names = dataset_names[1:]
    
    vprint(f"Training on: {train_name}")
    
    all_results = {}
    
    for test_name in test_names:
        vprint(f"Testing on: {test_name}")
        
        test_adata = datasets[test_name].copy()
        vprint(f"Test data (pre-processed): {test_adata.n_obs} cells × {test_adata.n_vars} genes")
        
        # Run ML analysis with transfer learning (original approach)
        output_dir = f'figures/machine_learning_results/{approach}/cross_dataset_{test_name}_fallback'
        
        # For transfer learning, we need original data
        train_adata_orig = datasets[train_name].copy()
        test_adata_orig = datasets[test_name].copy()
        
        # Preprocess training data for traditional ML
        train_adata_processed = preprocess_for_ml(datasets[train_name].copy())
        
        # Check if we have enough common genes for analysis
        common_genes = list(set(train_adata_processed.var_names) & set(test_adata.var_names))
        if len(common_genes) < 50:
            vprint(f"Too few common genes ({len(common_genes)}) for reliable analysis")
            vprint(f"Skipping {test_name} cross-dataset analysis")
            continue
        
        try:
            results, le = run_ml_analysis(train_adata_processed, test_adata, output_dir, 
                                        include_transfer_learning=True,
                                        train_adata_orig=train_adata_orig, 
                                        test_adata_orig=test_adata_orig)
        except Exception as e:
            vprint(f" Error in cross-dataset analysis for {test_name}: {str(e)[:100]}...")
            continue
        
        all_results[test_name] = results
    
    return all_results

def main():
    """Main ML pipeline function"""
    parser = argparse.ArgumentParser(description='Unified ML Pipeline for PBMC Data')
    parser.add_argument('--approach', choices=['annotated', 'not_annotated', 'both'], 
                       default='both', help='Which approach to run ML analysis for')
    parser.add_argument('--mode', choices=['same', 'cross', 'both'], 
                       default='both', help='Analysis mode')
    parser.add_argument('--verbose', '-v', action='store_true', default=True,
                        help='Enable verbose output (default: True)')
    parser.add_argument('--quiet', '-q', action='store_true', default=False,
                        help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Handle verbose/quiet flags
    verbose = args.verbose and not args.quiet
    set_verbose(verbose)
    
    vprint("Unified PBMC ML Pipeline")
    vprint("="* 40)
    vprint(f"Approach: {args.approach}")
    vprint(f"Mode: {args.mode}")
    
    success = True
    
    # Process each approach
    approaches_to_run = []
    if args.approach == 'both':
        approaches_to_run = ['not_annotated', 'annotated']
    else:
        approaches_to_run = [args.approach]
    
    for approach in approaches_to_run:
        vprint(f"{'='*60}")
        vprint(f"Processing {approach.upper()} approach")
        vprint('='*60)
        
        try:
            # Same-dataset analysis
            if args.mode in ['same', 'both']:
                same_results = run_same_dataset_analysis(approach)
                if same_results:
                    vprint(f"Same-dataset analysis completed for {approach}")
                else:
                    vprint(f"Same-dataset analysis failed for {approach}")
                    success = False
            
            # Cross-dataset analysis  
            if args.mode in ['cross', 'both']:
                cross_results = run_cross_dataset_analysis(approach)
                if cross_results:
                    vprint(f"Cross-dataset analysis completed for {approach}")
                else:
                    vprint(f"Cross-dataset analysis failed for {approach}")
                    success = False
                    
        except Exception as e:
            vprint(f"Error processing {approach}: {str(e)}")
            success = False
    
    if success:
        vprint("ML Pipeline Complete!")
        vprint("Results saved to:")
        vprint("figures/machine_learning_results/")
        vprint("Ready to run notebooks:")
        vprint("final_project_cell_classification.ipynb (not_annotated)")
        vprint("final_project_seuratdata_standalone.ipynb (annotated)")
    else:
        vprint("ML Pipeline completed with errors")
    
    return success

if __name__ == "__main__":
    main()
