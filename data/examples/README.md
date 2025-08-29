# Data Directory Structure

This directory will be populated when you run the data download script. The structure will be:

```
data/
├── annotated/                    # Expert-annotated datasets (SeuratData approach)
│   ├── pbmc3k_processed.h5ad
│   ├── pbmc_multiome_full_processed.h5ad
│   └── pbmc_cite_seq_processed.h5ad
├── not_annotated/               # Datasets requiring clustering-based annotation
│   ├── pbmc3k_processed.h5ad
│   ├── pbmc3k_full_processed.h5ad
│   ├── pbmc_multiome_processed.h5ad
│   ├── pbmc_multiome_full_processed.h5ad
│   ├── pbmc_cite_seq_processed.h5ad
│   └── pbmc_cite_seq_full_processed.h5ad
└── cache/                       # Temporary processing files
    └── [various intermediate files]
```

## Data Sources

All datasets are sourced from **10x Genomics** public datasets:
- **PBMC 3k**: Single Cell Gene Expression Dataset
- **PBMC Multiome**: Multiome ATAC + Gene Expression  
- **PBMC CITE-seq**: Single Cell Immune Profiling

## Download Instructions

Run the unified download script:
```bash
python scripts/download_data_unified.py --mode both --verbose
```

This will automatically:
1. Download raw data from 10x Genomics
2. Process and filter datasets  
3. Create both annotated and not_annotated versions
4. Save in standardized AnnData (.h5ad) format

## Data Size Note

**Warning**: Downloaded datasets can be several GB. Ensure adequate disk space before running the download script.

The `.gitignore` file excludes data files from version control due to their size.
