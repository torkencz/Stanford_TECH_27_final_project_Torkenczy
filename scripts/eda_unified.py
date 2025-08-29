#!/usr/bin/env python3
"""
Unified EDA Script for PBMC Single-Cell Classification Project
Handles both annotated (SeuratData) and not_annotated (10x) approaches
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
from scipy import sparse, io
from sklearn.decomposition import PCA
import umap
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

# Set high-quality plotting parameters
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

def add_cell_types_clustering_based(adata):
    """Add cell type labels using proper clustering-based annotation"""
    vprint(" Applying clustering-based cell type annotation...")
    
    # Known marker genes for each cell type (comprehensive list)
    marker_genes = {
        'T cells': {
            'CD4+ T': ['CD3D', 'CD3E', 'CD4', 'IL7R', 'CCR7', 'LEF1'],
            'CD8+ T': ['CD3D', 'CD3E', 'CD8A', 'CD8B', 'GZMK', 'CCL5'],
            'Naive CD4+ T': ['CD3D', 'CD4', 'IL7R', 'CCR7', 'TCF7', 'LEF1'],
            'Memory CD4+ T': ['CD3D', 'CD4', 'IL7R', 'CD44', 'CD69'],
            'Regulatory T': ['CD3D', 'CD4', 'FOXP3', 'IL2RA', 'CTLA4'],
            'NK T': ['CD3D', 'KLRD1', 'KLRF1', 'NCR1']
        },
        'B cells': {
            'Naive B': ['MS4A1', 'CD79A', 'CD79B', 'IGHD', 'TCL1A'],
            'Memory B': ['MS4A1', 'CD79A', 'CD27', 'IGHG1'],
            'Plasma': ['IGHG1', 'MZB1', 'SDC1', 'CD27', 'XBP1']
        },
        'Myeloid': {
            'CD14+ Monocytes': ['LYZ', 'CD14', 'S100A8', 'S100A9', 'FCN1'],
            'CD16+ Monocytes': ['LYZ', 'FCGR3A', 'MS4A7', 'CDKN1C'],
            'Dendritic': ['FCER1A', 'CST3', 'CLEC10A']
        },
        'NK cells': {
            'NK': ['NKG7', 'GNLY', 'KLRF1', 'KLRD1', 'NCAM1', 'FCGR3A']
        },
        'Other': {
            'Platelets': ['PPBP', 'PF4', 'GNG11']
        }
    }
    
    # Ensure we have the required preprocessing
    if 'leiden' not in adata.obs.columns:
        vprint("Running clustering...")
        # Run Leiden clustering
        sc.tl.leiden(adata, resolution=0.5, random_state=42)
    
    n_clusters = len(adata.obs['leiden'].unique())
    vprint(f"Found {n_clusters} clusters")
    
    # Find marker genes for each cluster
    vprint("Finding marker genes per cluster...")
    sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon', n_genes=20)
    
    # Get top marker genes for each cluster
    cluster_markers = {}
    for cluster in adata.obs['leiden'].unique():
        markers = adata.uns['rank_genes_groups']['names'][cluster][:10]  # Top 10 markers
        cluster_markers[cluster] = list(markers)
    
    # Annotate clusters based on marker gene expression
    def annotate_cluster(cluster_id, top_markers):
        """Annotate a single cluster based on its top marker genes"""
        
        # Calculate enrichment scores for each cell type
        scores = {}
        
        for main_type, subtypes in marker_genes.items():
            for subtype, markers in subtypes.items():
                score = 0
                for marker in markers:
                    if marker in top_markers:
                        # Weight by position in top markers (earlier = higher weight)
                        weight = (11 - top_markers.index(marker)) / 10.0 if marker in top_markers[:10] else 0
                        score += weight
                scores[subtype] = score
        
        # Find best matching cell type
        if max(scores.values()) > 0:
            best_type = max(scores, key=scores.get)
            confidence = scores[best_type] / sum(scores.values()) if sum(scores.values()) > 0 else 0
        else:
            best_type = 'Unknown'
            confidence = 0
            
        return best_type, confidence
    
    # Annotate each cluster
    cluster_annotations = {}
    vprint("Annotating clusters:")
    
    for cluster in adata.obs['leiden'].unique():
        top_markers = cluster_markers[cluster]
        cell_type, confidence = annotate_cluster(cluster, top_markers)
        cluster_annotations[cluster] = cell_type
        
        vprint(f"   Cluster {cluster}: {cell_type} (confidence: {confidence:.2f})")
        vprint(f"      Top markers: {', '.join(top_markers[:5])}")
    
    # Assign cell types to cells based on cluster annotations
    cell_types = adata.obs['leiden'].map(cluster_annotations).values
    
    vprint(f"Final cell type distribution:")
    unique_types, counts = np.unique(cell_types, return_counts=True)
    for cell_type, count in zip(unique_types, counts):
        percentage = (count / len(cell_types)) * 100
        vprint(f"   {cell_type}: {count} cells ({percentage:.1f}%)")
    
    return cell_types, cluster_annotations

def preprocess_for_ml(adata, min_genes=200, min_cells=3, n_top_genes=2000, mito_threshold=0.2):
    """Enhanced preprocessing for machine learning with mitochondrial filtering, PCA, and UMAP"""
    vprint(f"Preprocessing {adata.n_obs} cells × {adata.n_vars} genes...")
    
    # Ensure unique gene names
    adata.var_names_make_unique()
    
    # Calculate mitochondrial gene fraction
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    
    vprint(f"Mitochondrial genes found: {adata.var['mt'].sum()}")
    vprint(f"Mean mitochondrial fraction: {adata.obs['pct_counts_mt'].mean():.2f}%")
    
    # Filter cells and genes
    n_cells_before = adata.n_obs
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    
    # Filter cells with high mitochondrial fraction
    adata = adata[adata.obs.pct_counts_mt < mito_threshold * 100, :].copy()
    n_cells_after_mito = adata.n_obs
    
    # Remove cells with no cell type annotation
    adata = adata[~adata.obs['cell_type'].isna()].copy()
    adata = adata[adata.obs['cell_type'] != 'filtered'].copy()  # Remove filtered cells if present
    
    n_cells_final = adata.n_obs
    vprint(f"After gene/cell filtering: {n_cells_before} → {adata.n_obs} cells")
    vprint(f"After mitochondrial filtering ({mito_threshold*100}%): {n_cells_after_mito} cells")
    vprint(f"After annotation filtering: {n_cells_final} cells × {adata.n_vars} genes")
    
    # Store raw data
    adata.raw = adata
    
    # Normalize and log transform
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # Find highly variable genes
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=n_top_genes)
    
    # Store HVG information before filtering
    n_hvg = adata.var.highly_variable.sum()
    vprint(f"Highly variable genes identified: {n_hvg}")
    
    # Keep original data for visualization
    adata_full = adata.copy()
    
    # Filter to highly variable genes for ML
    adata = adata[:, adata.var.highly_variable].copy()
    
    # Scale data
    sc.pp.scale(adata, max_value=10)
    
    # Calculate PCA
    vprint(f"Computing PCA...")
    sc.tl.pca(adata, svd_solver='arpack', n_comps=50)
    
    # Calculate UMAP
    vprint(f"Computing neighbors and UMAP...")
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    sc.tl.umap(adata)
    
    # Store dimensionality reduction results in full data (only obs-level data)
    adata_full.obsm['X_pca'] = adata.obsm['X_pca']
    adata_full.obsm['X_umap'] = adata.obsm['X_umap']
    # Note: PCs and other var-level data stay with the filtered dataset since dimensions differ
    
    vprint(f"Final processed data: {adata.n_obs} cells × {adata.n_vars} genes")
    vprint(f"PCA explained variance ratio (first 10 PCs): {adata.uns['pca']['variance_ratio'][:10].sum():.3f}")
    
    return adata, adata_full

def load_10x_data(data_dir):
    """Load 10x PBMC data"""
    vprint(f" Loading 10x data from: {data_dir}")
    
    # Look for matrix files in the directory structure
    matrix_file = None
    genes_file = None
    barcodes_file = None
    
    # Search through directory structure
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file == 'matrix.mtx' or file == 'matrix.mtx.gz':
                matrix_file = os.path.join(root, file)
            elif file in ['genes.tsv', 'features.tsv', 'genes.tsv.gz', 'features.tsv.gz']:
                genes_file = os.path.join(root, file)
            elif file in ['barcodes.tsv', 'barcodes.tsv.gz']:
                barcodes_file = os.path.join(root, file)
    
    if not all([matrix_file, genes_file, barcodes_file]):
        raise FileNotFoundError(f"Could not find required 10x files in {data_dir}")
    
    # Load data
    X = io.mmread(matrix_file).T.tocsr()  # cells x genes
    
    # Load genes/features
    if genes_file.endswith('.gz'):
        import gzip
        with gzip.open(genes_file, 'rt') as f:
            genes_df = pd.read_csv(f, sep='\\t', header=None)
    else:
        genes_df = pd.read_csv(genes_file, sep='\\t', header=None)
    
    # Handle features vs genes format
    if genes_df.shape[1] >= 3:
        # Filter for Gene Expression if multimodal
        if 'Gene Expression' in genes_df.iloc[:, 2].values:
            rna_mask = genes_df.iloc[:, 2] == 'Gene Expression'
            X = X[:, rna_mask.values]
            genes = genes_df[rna_mask].iloc[:, 1].values
        else:
            genes = genes_df.iloc[:, 1].values
    else:
        genes = genes_df.iloc[:, 1].values
    
    # Load barcodes
    if barcodes_file.endswith('.gz'):
        import gzip
        with gzip.open(barcodes_file, 'rt') as f:
            barcodes = pd.read_csv(f, sep='\\t', header=None).iloc[:, 0].values
    else:
        barcodes = pd.read_csv(barcodes_file, sep='\\t', header=None).iloc[:, 0].values
    
    vprint(f"Loaded: {X.shape[0]} cells × {X.shape[1]} genes")
    return X, genes, barcodes

def load_annotated_data(data_dir):
    """Load annotated SeuratData"""
    vprint(f" Loading annotated data from: {data_dir}")
    
    expr_file = os.path.join(data_dir, 'expression.csv')
    meta_file = os.path.join(data_dir, 'metadata.csv')
    
    if not os.path.exists(expr_file) or not os.path.exists(meta_file):
        raise FileNotFoundError(f"Missing expression.csv or metadata.csv in {data_dir}")
    
    # Load expression data - handle SeuratData CSV format
    expr_df = pd.read_csv(expr_file)
    meta_df = pd.read_csv(meta_file, index_col=0)
    
    # Handle SeuratData CSV format where gene_id is the last column
    if 'gene_id' in expr_df.columns:
        # Set gene_id as index and remove it from columns
        genes = expr_df['gene_id'].values
        expr_df = expr_df.drop('gene_id', axis=1)
        expr_df.index = genes
    else:
        # Standard format with gene names as index
        genes = expr_df.index.values
    
    # Ensure all columns are numeric
    expr_df = expr_df.select_dtypes(include=[np.number])
    
    # Convert to matrix format
    X = sparse.csr_matrix(expr_df.T.values)  # cells x genes
    genes = expr_df.index.values
    barcodes = expr_df.columns.values
    
    vprint(f" Loaded: {X.shape[0]} cells × {X.shape[1]} genes")
    vprint(f"Existing annotations: {meta_df['seurat_annotations'].nunique()} cell types")
    
    return X, genes, barcodes, meta_df

def create_clustering_plots(adata, dataset_name, output_dir):
    """Create clustering-specific plots"""
    if 'leiden' not in adata.obs.columns:
        return
        
    vprint(f"Creating clustering plots...")
    
    # Create clustering visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: UMAP colored by clusters
    if 'X_umap' in adata.obsm:
        umap_coords = adata.obsm['X_umap']
        clusters = adata.obs['leiden'].astype('category')
        
        for cluster in clusters.cat.categories:
            mask = clusters == cluster
            ax1.scatter(umap_coords[mask, 0], umap_coords[mask, 1], 
                       s=10, alpha=0.7, label=f'Cluster {cluster}')
        
        ax1.set_xlabel('UMAP 1')
        ax1.set_ylabel('UMAP 2')
        ax1.set_title('Leiden Clusters')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 2: UMAP colored by cell types
    if 'X_umap' in adata.obsm and 'cell_type' in adata.obs:
        cell_types = adata.obs['cell_type'].astype('category')
        
        for cell_type in cell_types.cat.categories:
            mask = cell_types == cell_type
            ax2.scatter(umap_coords[mask, 0], umap_coords[mask, 1], 
                       s=10, alpha=0.7, label=cell_type)
        
        ax2.set_xlabel('UMAP 1')
        ax2.set_ylabel('UMAP 2')
        ax2.set_title('Cell Type Annotations')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 3: Cluster size distribution
    if 'leiden' in adata.obs:
        cluster_counts = adata.obs['leiden'].value_counts().sort_index()
        ax3.bar(range(len(cluster_counts)), cluster_counts.values, 
                color='lightblue', edgecolor='black')
        ax3.set_xlabel('Cluster')
        ax3.set_ylabel('Number of Cells')
        ax3.set_title('Cluster Size Distribution')
        ax3.set_xticks(range(len(cluster_counts)))
        ax3.set_xticklabels(cluster_counts.index)
    
    # Plot 4: Cell type distribution
    if 'cell_type' in adata.obs:
        type_counts = adata.obs['cell_type'].value_counts()
        ax4.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%',
                startangle=90)
        ax4.set_title('Cell Type Distribution')
    
    plt.tight_layout()
    
    # Save plots
    clustering_path = f'{output_dir}/{dataset_name}_clustering_analysis'
    plt.savefig(f'{clustering_path}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{clustering_path}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    vprint(f"Clustering plots saved to: {clustering_path}.pdf/png")

def create_eda_plots(adata, approach, dataset_name, output_dir):
    """Create comprehensive EDA plots"""
    vprint(f" Creating EDA plots for {dataset_name} ({approach})...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Figure 1: Basic statistics
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Number of genes per cell
    n_genes = (adata.X > 0).sum(axis=1).A1
    ax1.hist(n_genes, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Number of Genes per Cell')
    ax1.set_ylabel('Number of Cells')
    ax1.set_title('Gene Count Distribution')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Total UMI per cell
    total_umi = adata.X.sum(axis=1).A1
    ax2.hist(total_umi, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    ax2.set_xlabel('Total UMI per Cell')
    ax2.set_ylabel('Number of Cells')
    ax2.set_title('UMI Count Distribution')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Cell type distribution
    if 'cell_type' in adata.obs.columns:
        cell_counts = adata.obs['cell_type'].value_counts()
        bars = ax3.bar(range(len(cell_counts)), cell_counts.values, 
                      color=plt.cm.Set3(np.linspace(0, 1, len(cell_counts))))
        ax3.set_xticks(range(len(cell_counts)))
        ax3.set_xticklabels(cell_counts.index, rotation=45, ha='right')
        ax3.set_ylabel('Number of Cells')
        ax3.set_title('Cell Type Distribution')
        ax3.grid(True, alpha=0.3)
        
        # Add count labels on bars
        for bar, count in zip(bars, cell_counts.values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'No cell type information available', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=14)
        ax3.set_title('Cell Type Information')
    
    # Plot 4: Highly expressed genes
    gene_expression_sum = adata.X.sum(axis=0).A1
    top_genes_idx = np.argsort(gene_expression_sum)[-20:][::-1]
    top_genes = adata.var_names[top_genes_idx]
    top_expression = gene_expression_sum[top_genes_idx]
    
    bars = ax4.barh(range(len(top_genes)), top_expression, color='lightgreen', alpha=0.7)
    ax4.set_yticks(range(len(top_genes)))
    ax4.set_yticklabels(top_genes)
    ax4.set_xlabel('Total Expression')
    ax4.set_title('Top 20 Highly Expressed Genes')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{dataset_name}_basic_statistics.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/{dataset_name}_basic_statistics.pdf', bbox_inches='tight')
    plt.close()
    
    # Figure 2: Quality metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scatter plot: genes vs UMI
    ax1.scatter(n_genes, total_umi, alpha=0.6, s=1)
    ax1.set_xlabel('Number of Genes')
    ax1.set_ylabel('Total UMI')
    ax1.set_title('Genes vs UMI per Cell')
    ax1.grid(True, alpha=0.3)
    
    # Gene detection frequency
    gene_detection_freq = (adata.X > 0).sum(axis=0).A1 / adata.n_obs
    ax2.hist(gene_detection_freq, bins=50, alpha=0.7, color='orange', edgecolor='black')
    ax2.set_xlabel('Detection Frequency')
    ax2.set_ylabel('Number of Genes')
    ax2.set_title('Gene Detection Frequency')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{dataset_name}_quality_metrics.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/{dataset_name}_quality_metrics.pdf', bbox_inches='tight')
    plt.close()
    
    vprint(f" EDA plots saved to: {output_dir}")

def create_enhanced_eda_plots(adata_processed, adata_full, approach, dataset_name, output_dir):
    """Create enhanced EDA plots including PCA and UMAP"""
    os.makedirs(output_dir, exist_ok=True)
    
    vprint(f"Creating enhanced EDA plots for {dataset_name}...")
    
    # Figure 1: PCA and UMAP visualizations
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # PCA plot
    sc.pl.pca(adata_processed, color='cell_type', ax=ax1, show=False, frameon=False)
    ax1.set_title(f'{dataset_name}: PCA Colored by Cell Type')
    
    # UMAP plot
    sc.pl.umap(adata_processed, color='cell_type', ax=ax2, show=False, frameon=False)
    ax2.set_title(f'{dataset_name}: UMAP Colored by Cell Type')
    
    # PCA explained variance
    explained_var = adata_processed.uns['pca']['variance_ratio']
    ax3.plot(range(1, min(21, len(explained_var)+1)), explained_var[:20], 'bo-')
    ax3.set_xlabel('Principal Component')
    ax3.set_ylabel('Explained Variance Ratio')
    ax3.set_title('PCA Explained Variance')
    ax3.grid(True, alpha=0.3)
    
    # Mitochondrial fraction distribution
    ax4.hist(adata_full.obs['pct_counts_mt'], bins=50, alpha=0.7, color='red', edgecolor='black')
    ax4.axvline(20, color='black', linestyle='--', label='20% threshold')
    ax4.set_xlabel('Mitochondrial Gene Percentage')
    ax4.set_ylabel('Number of Cells')
    ax4.set_title('Mitochondrial Gene Fraction Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{dataset_name}_enhanced_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/{dataset_name}_enhanced_analysis.pdf', bbox_inches='tight')
    plt.close()
    
    # Figure 2: Quality control metrics
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Total counts vs mitochondrial percentage
    ax1.scatter(adata_full.obs['total_counts'], adata_full.obs['pct_counts_mt'], 
                alpha=0.6, s=1, c='blue')
    ax1.set_xlabel('Total UMI Counts')
    ax1.set_ylabel('Mitochondrial Gene Percentage')
    ax1.set_title('UMI vs Mitochondrial Content')
    ax1.grid(True, alpha=0.3)
    
    # Genes vs mitochondrial percentage
    ax2.scatter(adata_full.obs['n_genes_by_counts'], adata_full.obs['pct_counts_mt'], 
                alpha=0.6, s=1, c='green')
    ax2.set_xlabel('Number of Genes')
    ax2.set_ylabel('Mitochondrial Gene Percentage')
    ax2.set_title('Gene Count vs Mitochondrial Content')
    ax2.grid(True, alpha=0.3)
    
    # Cell type distribution
    cell_type_counts = adata_processed.obs['cell_type'].value_counts()
    colors = plt.cm.Set3(range(len(cell_type_counts)))
    ax3.pie(cell_type_counts.values, labels=cell_type_counts.index, autopct='%1.1f%%', colors=colors)
    ax3.set_title('Cell Type Distribution (After Filtering)')
    
    # Highly variable genes
    hvg_df = adata_full.var[adata_full.var.highly_variable].copy()
    if len(hvg_df) > 0:
        ax4.scatter(hvg_df['means'], hvg_df['dispersions_norm'], alpha=0.6, s=2)
        ax4.set_xlabel('Mean Expression')
        ax4.set_ylabel('Normalized Dispersion')
        ax4.set_title(f'Highly Variable Genes (n={len(hvg_df)})')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No HVG data available', transform=ax4.transAxes, 
                ha='center', va='center', fontsize=12)
        ax4.set_title('Highly Variable Genes')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{dataset_name}_quality_control.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/{dataset_name}_quality_control.pdf', bbox_inches='tight')
    plt.close()
    
    vprint(f"Enhanced EDA plots saved to: {output_dir}")

def process_dataset(approach, dataset_name, data_path):
    """Process a single dataset"""
    vprint(f"Processing {dataset_name} ({approach})")
    #print("=" * 50)
    
    if approach == 'not_annotated':
        # Load 10x data
        X, genes, barcodes = load_10x_data(data_path)
        
        # Create AnnData object
        adata = sc.AnnData(X=X)
        adata.var_names = genes
        adata.obs_names = barcodes
        
        # Apply initial preprocessing for clustering
        vprint("Performing initial preprocessing for clustering...")
        adata_temp = adata.copy()
        adata_temp.var_names_make_unique()
        
        # Store original cell barcodes before filtering
        original_barcodes = adata_temp.obs_names.copy()
        
        sc.pp.filter_cells(adata_temp, min_genes=200)
        sc.pp.filter_genes(adata_temp, min_cells=3)
        
        # Get barcodes of cells that passed filtering
        filtered_barcodes = adata_temp.obs_names.copy()
        
        adata_temp.raw = adata_temp
        sc.pp.normalize_total(adata_temp, target_sum=1e4)
        sc.pp.log1p(adata_temp)
        sc.pp.highly_variable_genes(adata_temp, min_mean=0.0125, max_mean=3, min_disp=0.5)
        adata_temp = adata_temp[:, adata_temp.var.highly_variable].copy()
        sc.pp.scale(adata_temp, max_value=10)
        sc.tl.pca(adata_temp, svd_solver='arpack')
        sc.pp.neighbors(adata_temp, n_neighbors=10, n_pcs=40)
        
        # Add proper clustering-based cell type annotation
        cell_types, cluster_annotations = add_cell_types_clustering_based(adata_temp)
        
        # Filter original adata to match filtered cells
        adata_filtered = adata[adata.obs_names.isin(filtered_barcodes)].copy()
        
        # Transfer annotations to filtered adata (matching dimensions)
        adata_filtered.obs['cell_type'] = cell_types
        adata_filtered.obs['leiden'] = adata_temp.obs['leiden'].values
        
        # Use the filtered data as our main adata
        adata = adata_filtered
        
        annotations_source = 'clustering_based'
        
    else:  # annotated
        # Load annotated data
        X, genes, barcodes, metadata = load_annotated_data(data_path)
        
        # Create AnnData object
        adata = sc.AnnData(X=X)
        adata.var_names = genes
        adata.obs_names = barcodes
        
        # Add expert annotations
        # Ensure same order
        common_barcodes = list(set(barcodes) & set(metadata.index))
        adata = adata[adata.obs_names.isin(common_barcodes)].copy()
        metadata_filtered = metadata.loc[adata.obs_names]
        
        adata.obs['cell_type'] = metadata_filtered['seurat_annotations'].values
        annotations_source = 'expert_curated'
    
    # Create basic EDA plots
    output_dir = f'figures/EDA/{approach}'
    create_eda_plots(adata, approach, dataset_name, output_dir)
    
    # Create clustering plots for clustering-based annotation
    if approach == 'not_annotated' and 'leiden' in adata.obs.columns:
        create_clustering_plots(adata, dataset_name, output_dir)
    
    # Apply ML preprocessing with enhanced features
    vprint(f"\n Applying enhanced preprocessing for {dataset_name}...")
    adata_processed, adata_full = preprocess_for_ml(adata.copy())
    
    # Create enhanced EDA plots with PCA/UMAP
    create_enhanced_eda_plots(adata_processed, adata_full, approach, dataset_name, output_dir)
    
    # Save both processed versions
    output_path = f'data/{approach}/{dataset_name}_processed.h5ad'
    adata_processed.write(output_path)
    vprint(f"ML-ready data saved to: {output_path}")
    
    # Save full version with all features
    full_output_path = f'data/{approach}/{dataset_name}_full_processed.h5ad'
    adata_full.write(full_output_path)
    vprint(f"Full processed data saved to: {full_output_path}")
    
    # Create enhanced summary
    summary = {
        'dataset': dataset_name,
        'approach': approach,
        'n_cells_raw': adata.n_obs,
        'n_genes_raw': adata.n_vars,
        'n_cells_processed': adata_processed.n_obs,
        'n_genes_processed': adata_processed.n_vars,
        'cells_filtered': adata.n_obs - adata_processed.n_obs,
        'genes_filtered': adata.n_vars - adata_processed.n_vars,
        'cell_types': adata_processed.obs['cell_type'].nunique(),
        'mito_genes_detected': adata_full.var['mt'].sum() if 'mt' in adata_full.var.columns else 0,
        'mean_mito_fraction': adata_full.obs['pct_counts_mt'].mean() if 'pct_counts_mt' in adata_full.obs.columns else 0,
        'annotations_source': annotations_source,
        'processed_file': output_path,
        'full_processed_file': full_output_path
    }
    
    return summary

def main():
    """Main EDA function"""
    parser = argparse.ArgumentParser(description='Unified EDA for PBMC Data')
    parser.add_argument('--approach', choices=['annotated', 'not_annotated', 'both'], 
                       default='both', help='Which approach to process')
    parser.add_argument('--verbose', '-v', action='store_true', default=True,
                        help='Enable verbose output (default: True)')
    parser.add_argument('--quiet', '-q', action='store_true', default=False,
                        help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Handle verbose/quiet flags
    verbose = args.verbose and not args.quiet
    set_verbose(verbose)
    
    vprint(" Unified PBMC EDA Pipeline")
    vprint(f" Processing approach: {args.approach}")
    
    summaries = []
    
    # Process not_annotated data
    if args.approach in ['not_annotated', 'both']:
        vprint(" Processing Not-Annotated Data (10x Genomics)")
        
        not_annotated_datasets = [
            ('pbmc3k', 'data/not_annotated/pbmc3k_extracted'),
            ('pbmc_multiome', 'data/not_annotated/pbmc_multiome_extracted'),
            ('pbmc_cite_seq', 'data/not_annotated/pbmc_cite_seq_extracted')
        ]
        
        for dataset_name, data_path in not_annotated_datasets:
            if os.path.exists(data_path):
                try:
                    summary = process_dataset('not_annotated', dataset_name, data_path)
                    summaries.append(summary)
                except Exception as e:
                    vprint(f" Failed to process {dataset_name}: {str(e)}")
            else:
                vprint(f"  Skipping {dataset_name}: {data_path} not found")
    
    # Process annotated data
    if args.approach in ['annotated', 'both']:
        vprint("Processing Annotated Data (SeuratData)")
        
        annotated_datasets = [
            ('pbmc3k', 'data/annotated/pbmc3k'),
            ('pbmc_multiome_full', 'data/annotated/pbmc_multiome_full')
        ]
        
        for dataset_name, data_path in annotated_datasets:
            if os.path.exists(data_path):
                try:
                    summary = process_dataset('annotated', dataset_name, data_path)
                    summaries.append(summary)
                except Exception as e:
                    vprint(f" Failed to process {dataset_name}: {str(e)}")
            else:
                vprint(f"  Skipping {dataset_name}: {data_path} not found")
    
    # Save EDA summary
    if summaries:
        summary_df = pd.DataFrame(summaries)
        summary_df.to_csv('data/eda_summary.csv', index=False)
        
        vprint("EDA Summary:")
        vprint(summary_df.to_string(index=False))
        vprint(f"EDA summary saved to: data/eda_summary.csv")
        
        vprint(" EDA Pipeline Complete!")
        vprint(" Ready to run:")
        vprint("python run_pipeline_unified.py --approach", args.approach)
        
        return True
    else:
        vprint(" No datasets processed successfully")
        return False

if __name__ == "__main__":
    main()
