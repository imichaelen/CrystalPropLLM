#!/usr/bin/env python3
"""
Custom CIF Preparation Script for CrystalPropLLM.

Prepare custom CIF files by processing them with pymatgen and saving to tar.gz archive.

Workflow:
1. Scan directory for CIF files
2. Parse and standardize each structure using pymatgen
3. Write standardized CIFs to tar.gz archive
4. Skip files with parsing errors

Usage:
  Basic preparation:
    python prepare_custom.py /path/to/cif_directory output.tar.gz
  
  Prepare CIFs from nested directories:
    python prepare_custom.py /path/to/root_dir all_cifs.tar.gz
"""

import os
import io
import tarfile
import argparse
from pymatgen.io.cif import CifWriter, Structure
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


def process_cif_files(input_dir, output_tar_gz):
    processed_count = 0
    error_count = 0
    
    with tarfile.open(output_tar_gz, "w:gz") as tar:
        for root, _, files in os.walk(input_dir):
            for file in tqdm(files, desc="preparing CIF files..."):
                if file.endswith(".cif"):
                    file_path = os.path.join(root, file)
                    try:
                        struct = Structure.from_file(file_path)
                        cif_content = CifWriter(struct=struct, symprec=0.1).__str__()

                        cif_file = tarfile.TarInfo(name=file)
                        cif_bytes = cif_content.encode("utf-8")
                        cif_file.size = len(cif_bytes)
                        tar.addfile(cif_file, io.BytesIO(cif_bytes))
                        processed_count += 1
                        
                    except (ValueError, Exception) as e:
                        print(f"Warning: Skipping {file} due to error: {e}")
                        error_count += 1
                        continue
    
    print(f"Successfully processed: {processed_count} files")
    print(f"Skipped due to errors: {error_count} files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare custom CIF files and save to a tar.gz file.")
    parser.add_argument("input_dir", help="Path to the directory containing CIF files.")
    parser.add_argument("output_tar_gz", help="Path to the output tar.gz file")
    args = parser.parse_args()

    process_cif_files(args.input_dir, args.output_tar_gz)

    print(f"Prepared CIF files have been saved to {args.output_tar_gz}")
