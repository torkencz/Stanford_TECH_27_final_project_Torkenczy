#!/usr/bin/env python3
"""
Transfer Learning Test on Full Datasets

This script tests transfer learning methods using the full datasets 
(all ~30,000+ genes) instead of the HVG-filtered versions to provide
a fair comparison with traditional ML methods.
"""

import numpy as np
import pandas as pd
import scanpy as sc
import warnings
warnings.filterwarnings('ignore')

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter
import sys
import os

# Add current directory to path
sys.path.append('.')

def load_full_datasets():
    """Load the full datasets (not HVG-filtered)"""
    print("Loading full datasets (all genes)...")
    
    # Load PBMC 3k (reference) - full dataset
    pbmc3k = sc.read_h5ad('data/not_annotated/pbmc3k_full_processed.h5ad')
    print(f"PBMC 3k: {pbmc3k.n_obs} cells, {pbmc3k.n_vars} genes")
    print(f"PBMC 3k cell types: {sorted(pbmc3k.obs['cell_type'].unique())}")
    
    # Load Multiome (query 1) - full dataset  
    multiome = sc.read_h5ad('data/not_annotated/pbmc_multiome_full_processed.h5ad')
    print(f"Multiome: {multiome.n_obs} cells, {multiome.n_vars} genes")
    print(f"Multiome cell types: {sorted(multiome.obs['cell_type'].unique())}")
    
    # Load CITE-seq (query 2) - full dataset
    cite_seq = sc.read_h5ad('data/not_annotated/pbmc_cite_seq_full_processed.h5ad')
    print(f"CITE-seq: {cite_seq.n_obs} cells, {cite_seq.n_vars} genes")
    print(f"CITE-seq cell types: {sorted(cite_seq.obs['cell_type'].unique())}")
    
    return pbmc3k, multiome, cite_seq

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
        'CD4+ T cells': 'CD4+ T',  # PBMC 3k naming
        'CD4+ T cell': 'CD4+ T',   # Alternative naming
        'CD4+ T': 'CD4+ T',        # Already harmonized
        
        'CD8 T': 'CD8+ T',
        'CD8 Naive': 'CD8+ T',
        'CD8 TEM_1': 'CD8+ T',
        'CD8 TEM_2': 'CD8+ T',
        'CD8+ T cell': 'CD8+ T',
        'CD8+ T': 'CD8+ T',         # Already harmonized
        
        # B cells
        'B': 'B',
        'Naive B': 'B',
        'Memory B': 'B', 
        'Intermediate B': 'B',
        'Plasma': 'B',
        'Naive B cells': 'B',  # PBMC 3k naming
        'B cell': 'B',         # Alternative naming
        
        # Monocytes
        'CD14+ Mono': 'Monocytes',
        'CD14 Mono': 'Monocytes',
        'FCGR3A+ Mono': 'Monocytes', 
        'CD16 Mono': 'Monocytes',
        'CD14+ Monocytes': 'Monocytes',  # PBMC 3k naming
        'CD14+ Monocyte': 'Monocytes',   # Alternative naming
        'CD16+ Monocytes': 'Monocytes',  # Additional type
        
        # NK cells
        'NK': 'NK',
        'NK cells': 'NK',  # PBMC 3k naming
        
        # Dendritic cells
        'DC': 'DC',
        'cDC': 'DC',
        'pDC': 'DC',
        'Dendritic cells': 'DC',  # PBMC 3k naming
        'Dendritic': 'DC',        # Short form
        
        # T cell subtypes
        'Naive CD4+ T': 'CD4+ T',    # Map to broader category
        'Regulatory T': 'CD4+ T',    # Tregs are CD4+ T cells
        
        # Other
        'Platelet': 'Other',
        'HSPC': 'Other',
        'MAIT': 'Other',
        'gdT': 'Other',
        'Platelets': 'Other',  # PBMC 3k naming
        'Unknown': 'Other',     # Map Unknown to Other
    }
    
    # Apply mapping
    y_train_mapped = np.array([label_mapping.get(label, 'Other') for label in y_train])
    y_test_mapped = np.array([label_mapping.get(label, 'Other') for label in y_test])
    
    return y_train_mapped, y_test_mapped

def run_knn_transfer_learning_full(adata_ref, adata_query, k=15):
    """Run k-NN based transfer learning on full datasets"""
    print(f"\n=== k-NN Transfer Learning on Full Data (k={k}) ===")
    
    # Get data
    X_ref = adata_ref.X.toarray() if hasattr(adata_ref.X, 'toarray') else adata_ref.X
    X_query = adata_query.X.toarray() if hasattr(adata_query.X, 'toarray') else adata_query.X
    y_ref = adata_ref.obs['cell_type'].values
    y_query = adata_query.obs['cell_type'].values
    
    print(f"Reference: {X_ref.shape[0]} cells, {X_ref.shape[1]} genes")
    print(f"Query: {X_query.shape[0]} cells, {X_query.shape[1]} genes")
    print(f"Reference types: {sorted(set(y_ref))}")
    print(f"Query types: {sorted(set(y_query))}")
    
    # Harmonize labels
    y_ref_harm, y_query_harm = harmonize_cell_types(y_ref, y_query)
    print(f"After harmonization:")
    print(f"Reference types: {sorted(set(y_ref_harm))}")
    print(f"Query types: {sorted(set(y_query_harm))}")
    
    # Find common genes
    ref_genes = adata_ref.var_names
    query_genes = adata_query.var_names
    common_genes = ref_genes.intersection(query_genes)
    print(f"Common genes: {len(common_genes)} out of {len(ref_genes)} ref, {len(query_genes)} query")
    
    if len(common_genes) < 1000:
        print(f"Warning: Only {len(common_genes)} common genes available")
    
    # Get indices for common genes
    ref_gene_indices = [i for i, gene in enumerate(ref_genes) if gene in common_genes]
    query_gene_indices = [i for i, gene in enumerate(query_genes) if gene in common_genes]
    
    # Subset data to common genes
    X_ref_common = X_ref[:, ref_gene_indices]
    X_query_common = X_query[:, query_gene_indices]
    
    print(f"After gene filtering: {X_ref_common.shape[1]} common genes")
    
    # Preprocess data
    print("Preprocessing data...")
    scaler = StandardScaler()
    X_ref_scaled = scaler.fit_transform(X_ref_common)
    X_query_scaled = scaler.transform(X_query_common)
    
    # Handle NaN/inf values
    X_ref_scaled = np.nan_to_num(X_ref_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    X_query_scaled = np.nan_to_num(X_query_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Train k-NN
    print("Training k-NN classifier...")
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    knn.fit(X_ref_scaled, y_ref_harm)
    
    # Predict
    print("Making predictions...")
    y_pred = knn.predict(X_query_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_query_harm, y_pred)
    precision = precision_score(y_query_harm, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_query_harm, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_query_harm, y_pred, average='weighted', zero_division=0)
    
    print(f"Results:")
    print(f"  Accuracy: {accuracy:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1-Score: {f1:.3f}")
    
    # Show prediction distribution
    pred_counts = Counter(y_pred)
    true_counts = Counter(y_query_harm)
    print(f"True label distribution: {dict(true_counts)}")
    print(f"Predicted label distribution: {dict(pred_counts)}")
    
    return accuracy, y_pred, y_query_harm, len(common_genes)

def run_scanpy_ingest_full(adata_ref, adata_query, use_rep='X_pca', k=50):
    """Run Scanpy ingest transfer learning on full datasets"""
    print(f"\n=== Scanpy Ingest Transfer Learning on Full Data ===")
    
    # Make copies
    adata_ref_copy = adata_ref.copy()
    adata_query_copy = adata_query.copy()
    
    # Get original labels
    y_ref = adata_ref_copy.obs['cell_type'].values
    y_query = adata_query_copy.obs['cell_type'].values
    
    print(f"Reference: {adata_ref_copy.n_obs} cells, {adata_ref_copy.n_vars} genes")
    print(f"Query: {adata_query_copy.n_obs} cells, {adata_query_copy.n_vars} genes")
    print(f"Reference types: {sorted(set(y_ref))}")
    print(f"Query types: {sorted(set(y_query))}")
    
    # Harmonize labels
    y_ref_harm, y_query_harm = harmonize_cell_types(y_ref, y_query)
    adata_ref_copy.obs['cell_type'] = y_ref_harm
    adata_query_copy.obs['cell_type'] = y_query_harm
    
    print(f"After harmonization:")
    print(f"Reference types: {sorted(set(y_ref_harm))}")
    print(f"Query types: {sorted(set(y_query_harm))}")
    
    # Find common genes
    common_genes = adata_ref_copy.var_names.intersection(adata_query_copy.var_names)
    print(f"Common genes: {len(common_genes)}")
    
    if len(common_genes) < 1000:
        print(f"Warning: Only {len(common_genes)} common genes available")
    
    # Subset to common genes
    adata_ref_common = adata_ref_copy[:, common_genes].copy()
    adata_query_common = adata_query_copy[:, common_genes].copy()
    
    # Compute PCA with more components for full data
    n_pca_components = min(50, len(common_genes)-1, adata_ref_common.n_obs-1)
    print(f"Computing PCA with {n_pca_components} components...")
    sc.tl.pca(adata_ref_common, n_comps=n_pca_components, svd_solver='arpack')
    sc.tl.pca(adata_query_common, n_comps=n_pca_components, svd_solver='arpack')
    
    # Compute neighbors for reference
    print(f"Computing neighborhood graph...")
    sc.pp.neighbors(adata_ref_common, n_neighbors=k, use_rep='X_pca', random_state=42)
    
    # Compute UMAP for reference
    print(f"Computing UMAP...")
    sc.tl.umap(adata_ref_common, random_state=42)
    
    # Apply ingest
    print(f"Applying sc.tl.ingest...")
    sc.tl.ingest(adata_query_common, adata_ref_common, obs='cell_type')
    
    # Get predictions
    y_pred = adata_query_common.obs['cell_type'].values
    
    # Calculate metrics
    accuracy = accuracy_score(y_query_harm, y_pred)
    precision = precision_score(y_query_harm, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_query_harm, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_query_harm, y_pred, average='weighted', zero_division=0)
    
    print(f"Results:")
    print(f"  Accuracy: {accuracy:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1-Score: {f1:.3f}")
    
    # Show prediction distribution
    pred_counts = Counter(y_pred)
    true_counts = Counter(y_query_harm)
    print(f"True label distribution: {dict(true_counts)}")
    print(f"Predicted label distribution: {dict(pred_counts)}")
    
    return accuracy, y_pred, y_query_harm, len(common_genes)

def main():
    """Main function to run transfer learning tests on full datasets"""
    print("=" * 80)
    print("TRANSFER LEARNING TEST ON FULL DATASETS")
    print("Testing with all available genes (not HVG-filtered)")
    print("=" * 80)
    
    # Load full data
    pbmc3k, multiome, cite_seq = load_full_datasets()
    
    # Test 1: PBMC 3k → Multiome (Full Data)
    print("\n" + "=" * 80)
    print("TEST 1: PBMC 3k → Multiome (Full Datasets)")
    print("=" * 80)
    
    # k-NN Transfer Learning
    knn_acc_multiome, _, _, knn_genes_multiome = run_knn_transfer_learning_full(pbmc3k, multiome)
    
    # Scanpy Ingest
    scanpy_acc_multiome, _, _, scanpy_genes_multiome = run_scanpy_ingest_full(pbmc3k, multiome)
    
    # Test 2: PBMC 3k → CITE-seq (Full Data)
    print("\n" + "=" * 80)
    print("TEST 2: PBMC 3k → CITE-seq (Full Datasets)")
    print("=" * 80)
    
    # k-NN Transfer Learning
    knn_acc_cite, _, _, knn_genes_cite = run_knn_transfer_learning_full(pbmc3k, cite_seq)
    
    # Scanpy Ingest
    scanpy_acc_cite, _, _, scanpy_genes_cite = run_scanpy_ingest_full(pbmc3k, cite_seq)
    
    # Summary
    print("\n" + "=" * 80)
    print("COMPARISON: HVG-FILTERED vs FULL DATASETS")
    print("=" * 80)
    
    print(f"PBMC 3k → Multiome:")
    print(f"  HVG-filtered (148 genes):")
    print(f"    k-NN Transfer Learning: 0.662")
    print(f"    Scanpy Ingest: 0.467")
    print(f"  Full datasets ({knn_genes_multiome} genes):")
    print(f"    k-NN Transfer Learning: {knn_acc_multiome:.3f}" if knn_acc_multiome else "    k-NN Transfer Learning: Failed")
    print(f"    Scanpy Ingest: {scanpy_acc_multiome:.3f}" if scanpy_acc_multiome else "    Scanpy Ingest: Failed")
    
    print(f"\nPBMC 3k → CITE-seq:")
    print(f"  HVG-filtered (334 genes):")
    print(f"    k-NN Transfer Learning: 0.901")
    print(f"    Scanpy Ingest: 0.745")
    print(f"  Full datasets ({knn_genes_cite} genes):")
    print(f"    k-NN Transfer Learning: {knn_acc_cite:.3f}" if knn_acc_cite else "    k-NN Transfer Learning: Failed")
    print(f"    Scanpy Ingest: {scanpy_acc_cite:.3f}" if scanpy_acc_cite else "    Scanpy Ingest: Failed")
    
    print(f"\nTraditional ML (Union HVG strategy): 2,947 genes")
    print(f"  PBMC 3k → Multiome: ~97.5% accuracy")
    print(f"  PBMC 3k → CITE-seq: ~98.9% accuracy")
    
    print("\nConclusion: This shows the impact of feature availability on transfer learning performance.")

if __name__ == "__main__":
    main()
