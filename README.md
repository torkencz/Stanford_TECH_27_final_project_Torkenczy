# Stanford TECH 27 Final Project: Advanced Single-Cell RNA-seq Analysis
**Author: Torkenczy**  
**Course: TECH 27 - Machine Learning for Bioinformatics**

## 🧬 Project Overview

This project presents a comprehensive **single-cell RNA sequencing (scRNA-seq) analysis pipeline** for cell type classification across multiple datasets. The work demonstrates advanced machine learning techniques applied to high-dimensional biological data, with a novel **Union Highly Variable Genes (HVG) strategy** that significantly outperforms existing transfer learning approaches.

For summary of results please download the slides, where I also recorded my voice to explain.

## 📊 Datasets

**Primary Datasets (10x Genomics PBMC)**:
- **PBMC 3k**: Training dataset (2,698 cells, 13,714 genes)
- **PBMC Multiome**: Cross-validation dataset (11,621 cells, 26,341 genes) 
- **PBMC CITE-seq**: Cross-validation dataset (7,798 cells, 18,054 genes)

**Cell Types Analyzed**: CD4+ T cells, CD8+ T cells, B cells, Monocytes, NK cells, Dendritic cells, Others

## 🚀 Methodology

### **1. Data Preprocessing & Quality Control**
- **Mitochondrial gene filtering** (< 20% MT content)
- **Cell and gene quality filtering** 
- **Normalization and log-transformation**
- **Batch effect assessment**

### **2. Cell Type Annotation**
- **Marker-based approach**: Simplified marker gene expression
- **Expert annotation approach**: Pre-annotated datasets via SeuratData
- **Clustering-based approach**: Leiden clustering + differential expression + automated marker enrichment

### **3. Feature Engineering**
- **Union HVG Strategy**: Novel approach identifying 2,947 common highly variable genes across datasets
- **PCA dimensionality reduction** (50 components)
- **Cross-dataset gene harmonization**

### **4. Machine Learning Pipeline**
**Traditional ML Algorithms (11 total)**:
- **Tree-based**: Random Forest, Gradient Boosting, Decision Tree, AdaBoost
- **Linear**: Logistic Regression, SVM (RBF kernel)
- **Instance-based**: K-Nearest Neighbors
- **Probabilistic**: Naive Bayes  
- **Neural Networks**: sklearn MLP, Keras MLP, Keras 1D CNN

**Transfer Learning Methods**:
- **k-NN Transfer Learning** (custom implementation)
- **Scanpy Ingest** (state-of-the-art manifold mapping)

### **5. Evaluation Metrics**
- Accuracy, Precision, Recall, F1-Score
- Specificity, ROC-AUC, PR-AUC
- Confusion matrices and per-class performance

## 📁 Repository Structure

```
TECH_27_final_project_Torkenczy/
├── README.md                          # This file
├── scripts/                          # Core pipeline scripts
│   ├── download_data_unified.py      # Data download and setup
│   ├── eda_unified.py               # Exploratory data analysis
│   ├── run_pipeline_unified.py      # Main ML pipeline
│   └── test_transfer_learning_full_data.py  # Transfer learning analysis
├── notebooks/                        # Jupyter notebooks
│   ├── final_project_marker_based.ipynb     # Marker-based approach
│   └── final_project_expert_annotations.ipynb # Expert annotation approach
├── reports/                          # Final reports and documentation
│   ├── final_project_marker_based.pdf       # Main project report (PDF)
│   └── UNIFIED_PIPELINE_SUMMARY.md          # Technical pipeline summary
├── figures/                          # Generated visualizations
│   ├── Mr_pipeline.png              # Pipeline overview diagram
│   ├── EDA/                         # Exploratory data analysis figures
│   └── machine_learning_results/    # ML performance visualizations
└── data/                            # Data directory structure (empty - see setup)
    └── examples/                    # Example data structure
```

## 🛠️ Setup & Installation

### **Prerequisites**
```bash
# Python 3.8+
pip install numpy pandas scikit-learn
pip install scanpy anndata
pip install matplotlib seaborn plotly
pip install tensorflow keras  # For deep learning models
pip install jupyter notebook  # For running notebooks

# R dependencies (for expert annotation approach)
# Install R 4.0+ and Bioconductor packages:
# Seurat, SeuratData, Signac
```

### **Quick Start**
```bash
# 1. Clone repository
git clone https://github.com/torkencz/Stanford_TECH_27_final_project_Torkenczy.git
cd Stanford_TECH_27_final_project_Torkenczy

# 2. Download data (creates data/ subdirectories)
python scripts/download_data_unified.py --mode both --verbose

# 3. Run exploratory data analysis  
python scripts/eda_unified.py --mode not_annotated --verbose

# 4. Run machine learning pipeline
python scripts/run_pipeline_unified.py --mode not_annotated --analysis both --verbose

# 5. View results in notebooks/
jupyter notebook notebooks/final_project_marker_based.ipynb
jupyter notebook notebooks/final_project_expert_annotation.ipynb
```

### **Command Line Options**
All scripts support:
- `--mode`: `annotated`, `not_annotated`, or `both`
- `--analysis`: `same_dataset`, `cross_dataset`, or `both` (ML pipeline only)
- `--verbose` / `--quiet`: Control output verbosity

## 📈 Performance Insights

**Key Findings**:
1. **Feature curation beats algorithm sophistication**: Union HVG (97-99%) >> Transfer learning (54-88%)
2. **Traditional ML excels with proper preprocessing**: Careful feature selection enables superior generalization
3. **Cross-dataset success**: Models generalize excellently across different single-cell technologies
4. **Biological interpretability**: Union HVG strategy selects biologically meaningful genes

## 📚 Usage Examples

### **Basic Analysis Pipeline**
```bash
# Complete analysis workflow
python scripts/download_data_unified.py --mode not_annotated
python scripts/eda_unified.py --mode not_annotated  
python scripts/run_pipeline_unified.py --mode not_annotated --analysis both
```

### **Transfer Learning Comparison**
```bash
# Test transfer learning on full datasets
python scripts/test_transfer_learning_full_data.py
```

### **Expert Annotation Approach**
```bash
# Using pre-annotated datasets
python scripts/download_data_unified.py --mode annotated
python scripts/eda_unified.py --mode annotated
python scripts/run_pipeline_unified.py --mode annotated --analysis cross_dataset

#View results in notebooks/

jupyter notebook notebooks/final_project_expert_annotation.ipynb
```

## 📞 Contact

**Author**: Kristof Torkenczy
**Course**: TECH 27 - Machine Learning for Bioinformatics  
 

For questions about methodology, implementation, or results, please open an issue in this repository.

