#!/usr/bin/env python3
"""
Unified Data Download Script for PBMC Single-Cell Classification Project
Downloads and organizes data for both annotated (SeuratData) and not_annotated (10x) approaches
"""

import os
import requests
import tarfile
import subprocess
import shutil
import argparse
import pandas as pd
from pathlib import Path

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

def create_directory_structure(verbose=True):
    """Create organized directory structure"""
    dirs = [
        'data/annotated',
        'data/not_annotated', 
        'figures/EDA/annotated',
        'figures/EDA/not_annotated',
        'figures/machine_learning_results/annotated',
        'figures/machine_learning_results/not_annotated'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    if verbose:
        vprint("📁 Created directory structure:")
        for dir_path in dirs:
            vprint(f" {dir_path}/")

def download_file(url, filename, verbose=True):
    """Download file with progress"""
    if verbose:
        vprint(f" Downloading {filename}...")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0 and verbose:
                        percent = (downloaded / total_size) * 100
                        vprint(f"\\r   Progress: {percent:.1f}%", end='', flush=True)
        
        if verbose:
            vprint(f"\\n Downloaded: {filename} ({downloaded/1024/1024:.1f} MB)")
        return True
        
    except Exception as e:
        if verbose:
            vprint(f"\\n    Download failed: {str(e)}")
        return False

def extract_tar_gz(filename, extract_dir, verbose=True):
    """Extract tar.gz file"""
    if verbose:
        vprint(f" Extracting {filename}...")
    
    try:
        with tarfile.open(filename, 'r:gz') as tar:
            tar.extractall(extract_dir)
        if verbose:
            vprint(f" Extracted to: {extract_dir}")
        return True
    except Exception as e:
        if verbose:
            vprint(f"    Extraction failed: {str(e)}")
        return False

def download_10x_data(verbose=True):
    """Download 10x Genomics datasets for not_annotated approach"""
    if verbose:
        vprint("\\n Downloading 10x Genomics PBMC Datasets (Not Annotated)")
        vprint("=" * 60)
    
    datasets = [
        {
            'name': 'pbmc3k',
            'url': 'https://cf.10xgenomics.com/samples/cell/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz',
            'filename': 'pbmc3k_filtered_gene_bc_matrices.tar.gz',
            'description': 'PBMC 3k cells (training data)'
        },
        {
            'name': 'pbmc_multiome',
            'url': 'https://cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_granulocyte_sorted_10k/pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.tar.gz',
            'filename': 'pbmc_multiome_test.tar.gz',
            'description': 'PBMC Multiome 10k cells (test data)'
        },
        {
            'name': 'pbmc_cite_seq',
            'url': 'https://cf.10xgenomics.com/samples/cell-exp/3.0.0/pbmc_10k_protein_v3/pbmc_10k_protein_v3_filtered_feature_bc_matrix.tar.gz',
            'filename': 'pbmc_cite_seq_test.tar.gz', 
            'description': 'PBMC 10k with protein (CITE-seq test data)'
        }
    ]
    
    success_count = 0
    
    for dataset in datasets:
        vprint(f"\\n Processing {dataset['description']}...")
        
        # Download
        if download_file(dataset['url'], dataset['filename'], verbose):
            # Extract to not_annotated folder
            extract_dir = f"data/not_annotated/{dataset['name']}_extracted"
            if extract_tar_gz(dataset['filename'], extract_dir, verbose):
                success_count += 1
                
                # Clean up tar file
                os.remove(dataset['filename'])
                vprint(f"Cleaned up: {dataset['filename']}")
            else:
                vprint(f"Failed to extract {dataset['filename']}")
        else:
            vprint(f"    Failed to download {dataset['name']}")
    
    vprint(f"\\n✨ 10x Downloads Complete: {success_count}/{len(datasets)} successful")
    return success_count == len(datasets)

def run_seuratdata_download(verbose=True):
    """Run SeuratData R scripts to download annotated data"""
    if verbose:
        vprint("\\n🔬 Running SeuratData Download (Expert Annotations)")
        vprint("=" * 60)
    
    # Check if R scripts exist
    scripts_needed = ["direct_pbmc3k_export.R", "full_multiome_export.R"]
    missing_scripts = [s for s in scripts_needed if not os.path.exists(s)]

    if missing_scripts:
        vprint(f" Missing SeuratData export scripts: {missing_scripts}")
        return False

    try:
        vprint("Running SeuratData export scripts...")

        # Export pbmc3k
        result1 = subprocess.run(['Rscript', 'direct_pbmc3k_export.R'],
                               capture_output=True, text=True, timeout=300)

        # Export full multiome RNA
        result2 = subprocess.run(['Rscript', 'full_multiome_export.R'],
                               capture_output=True, text=True, timeout=900)

        vprint("PBMC3k export output:")
        vprint(result1.stdout)
        if result1.stderr:
            vprint("PBMC3k warnings:")
            vprint(result1.stderr)

        vprint("\\nFull Multiome RNA export output:")
        vprint(result2.stdout)
        if result2.stderr:
            vprint("Full Multiome warnings:")
            vprint(result2.stderr)

        success = (result1.returncode == 0) and (result2.returncode == 0)
        
        if success:
            # Move exported files to data/annotated/
            vprint("\\nMoving SeuratData exports to data/annotated/...")
            
            # Create subdirectories
            os.makedirs('data/annotated/pbmc3k', exist_ok=True)
            os.makedirs('data/annotated/pbmc_multiome_full', exist_ok=True)
            
            # Move files if they exist
            seuratdata_files = [
                ('seuratdata/pbmc3k_expression.csv', 'data/annotated/pbmc3k/expression.csv'),
                ('seuratdata/pbmc3k_metadata.csv', 'data/annotated/pbmc3k/metadata.csv'),
                ('seuratdata/pbmc_multiome_full_expression.csv', 'data/annotated/pbmc_multiome_full/expression.csv'),
                ('seuratdata/pbmc_multiome_full_metadata.csv', 'data/annotated/pbmc_multiome_full/metadata.csv'),
            ]
            
            moved_count = 0
            for src, dst in seuratdata_files:
                if os.path.exists(src):
                    shutil.move(src, dst)
                    vprint(f" Moved: {src} → {dst}")
                    moved_count += 1
                else:
                    vprint(f"     File not found: {src}")
            
            # Remove empty seuratdata directory if it exists
            if os.path.exists('seuratdata') and not os.listdir('seuratdata'):
                os.rmdir('seuratdata')
                vprint("     Removed empty seuratdata directory")
            
            vprint(f"\\n✨ SeuratData Download Complete: {moved_count}/{len(seuratdata_files)} files moved")
            return True
        else:
            vprint("\\n SeuratData export failed")
            return False

    except subprocess.TimeoutExpired:
        vprint("\\n SeuratData export timed out")
        return False
    except Exception as e:
        vprint(f"\\n SeuratData export error: {str(e)}")
        return False

def create_data_summary():
    """Create summary of downloaded data"""
    vprint("\\n Creating Data Summary...")
    
    summary = {
        'dataset': [],
        'approach': [],
        'path': [],
        'status': []
    }
    
    # Check not_annotated data
    not_annotated_datasets = ['pbmc3k', 'pbmc_multiome', 'pbmc_cite_seq']
    for dataset in not_annotated_datasets:
        path = f"data/not_annotated/{dataset}_extracted"
        status = " Available" if os.path.exists(path) else " Missing"
        
        summary['dataset'].append(dataset)
        summary['approach'].append('not_annotated')
        summary['path'].append(path)
        summary['status'].append(status)
    
    # Check annotated data
    annotated_datasets = ['pbmc3k', 'pbmc_multiome_full']
    for dataset in annotated_datasets:
        path = f"data/annotated/{dataset}"
        status = " Available" if os.path.exists(path) else " Missing"
        
        summary['dataset'].append(dataset)
        summary['approach'].append('annotated')
        summary['path'].append(path)
        summary['status'].append(status)
    
    # Save summary
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv('data/data_summary.csv', index=False)
    
    vprint("\\n Data Summary:")
    vprint(summary_df.to_string(index=False))
    vprint(f"\\n Summary saved to: data/data_summary.csv")
    
    return summary_df

def main():
    """Main download function"""
    parser = argparse.ArgumentParser(description='Unified PBMC Data Download')
    parser.add_argument('--approach', choices=['annotated', 'not_annotated', 'both'], 
                       default='both', help='Which approach to download data for')
    parser.add_argument('--verbose', '-v', action='store_true', default=True,
                        help='Enable verbose output (default: True)')
    parser.add_argument('--quiet', '-q', action='store_true', default=False,
                        help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Handle verbose/quiet flags
    verbose = args.verbose and not args.quiet
    set_verbose(verbose)
    
    vprint(" Unified PBMC Data Download Pipeline")
    vprint("=" * 50)
    vprint(f" Download approach: {args.approach}")
    
    # Create directory structure
    create_directory_structure(verbose)
    
    success_10x = True
    success_seuratdata = True
    
    # Download based on approach
    if args.approach in ['not_annotated', 'both']:
        success_10x = download_10x_data(verbose)
    
    if args.approach in ['annotated', 'both']:
        success_seuratdata = run_seuratdata_download(verbose)
    
    # Create data summary
    summary_df = create_data_summary()
    
    # Final status
    vprint("\\n Download Pipeline Complete!")
    
    if args.approach == 'both':
        if success_10x and success_seuratdata:
            vprint(" Both approaches downloaded successfully")
            vprint("\\n Ready to run:")
            vprint("   python eda_unified.py --approach both")
        else:
            vprint("  Some downloads failed - check logs above")
    elif args.approach == 'not_annotated' and success_10x:
        vprint(" 10x data downloaded successfully")  
        vprint("\\n Ready to run:")
        vprint("   python eda_unified.py --approach not_annotated")
    elif args.approach == 'annotated' and success_seuratdata:
        vprint(" SeuratData downloaded successfully")
        vprint("\\n Ready to run:")
        vprint("   python eda_unified.py --approach annotated")
    else:
        vprint(" Download failed")
        return False
    
    return True

if __name__ == "__main__":
    main()
