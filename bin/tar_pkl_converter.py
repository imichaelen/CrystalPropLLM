#!/usr/bin/env python3
"""
CIF Data Format Converter for CrystalPropLLM.

Convert CIF data between .tar.gz and .pkl.gz formats in both directions.

Workflow:
1. Auto-detect or specify conversion mode
2. Load data from source format
3. Convert to target format
4. Save converted data

Usage:
  Convert tar to pickle (auto-detected):
    python tar_pkl_converter.py data.tar.gz data.pkl.gz
  
  Convert pickle to tar (auto-detected):
    python tar_pkl_converter.py data.pkl.gz data.tar.gz
  
  Explicit mode specification:
    python tar_pkl_converter.py --mode tar_to_pickle data.tar.gz data.pkl.gz
"""

import os
import tarfile
import argparse
import pickle
import gzip
import io
from tqdm import tqdm


def tar_to_pickle(tar_gz_filename):
    """Load CIF data from a .tar.gz file and return as a list of (id, content) tuples."""
    print(f"Loading CIF data from {tar_gz_filename}...")
    cif_data = []
    
    with tarfile.open(tar_gz_filename, "r:gz") as tar:
        for member in tqdm(tar.getmembers(), desc="Extracting files"):
            f = tar.extractfile(member)
            if f is not None:
                content = f.read().decode("utf-8")
                filename = os.path.basename(member.name)
                cif_id = filename.replace(".cif", "")
                cif_data.append((cif_id, content))
    
    return cif_data


def pickle_to_tar(cif_data, tar_gz_filename):
    """Save CIF data as a .tar.gz file."""
    print(f"Saving CIF data to {tar_gz_filename}...")
    
    with tarfile.open(tar_gz_filename, "w:gz") as tar:
        for item_id, cif_content in tqdm(cif_data, desc="Creating tar archive"):
            cif_file = io.BytesIO(cif_content.encode("utf-8"))
            tarinfo = tarfile.TarInfo(name=f"{item_id}.cif")
            tarinfo.size = len(cif_file.getvalue())
            cif_file.seek(0)
            tar.addfile(tarinfo, cif_file)


def load_pickle(pickle_gz_filename):
    """Load CIF data from a .pkl.gz file."""
    print(f"Loading CIF data from {pickle_gz_filename}...")
    
    with gzip.open(pickle_gz_filename, "rb") as f:
        cif_data = pickle.load(f)
    
    return cif_data


def save_pickle(cif_data, pickle_gz_filename):
    """Save CIF data to a .pkl.gz file."""
    print(f"Saving CIF data to {pickle_gz_filename}...")
    
    with gzip.open(pickle_gz_filename, "wb") as f:
        pickle.dump(cif_data, f, protocol=pickle.HIGHEST_PROTOCOL)


def detect_conversion_mode(input_file, output_file):
    """Automatically detect conversion mode based on file extensions."""
    input_ext = get_file_type(input_file)
    output_ext = get_file_type(output_file)
    
    if input_ext == "tar" and output_ext == "pickle":
        return "tar_to_pickle"
    elif input_ext == "pickle" and output_ext == "tar":
        return "pickle_to_tar"
    else:
        raise ValueError(f"Cannot determine conversion mode from {input_ext} to {output_ext}")


def get_file_type(filename):
    """Determine file type based on extension."""
    if filename.endswith((".tar.gz", ".tgz")):
        return "tar"
    elif filename.endswith((".pkl.gz", ".pickle.gz")):
        return "pickle"
    else:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(
        description="Convert CIF data between .tar.gz and .pkl.gz formats."
    )
    
    parser.add_argument("input", help="Path to the input file")
    parser.add_argument("output", help="Path to the output file")
    parser.add_argument(
        "--mode", 
        choices=["tar_to_pickle", "pickle_to_tar"],
        help="Explicitly specify conversion mode (auto-detected if not provided)"
    )
    
    args = parser.parse_args()
    
    # Determine conversion mode
    if args.mode:
        mode = args.mode
    else:
        try:
            mode = detect_conversion_mode(args.input, args.output)
        except ValueError as e:
            parser.error(str(e))
    
    # Perform conversion
    try:
        if mode == "tar_to_pickle":
            print(f"Converting {args.input} (tar) → {args.output} (pickle)")
            cif_data = tar_to_pickle(args.input)
            save_pickle(cif_data, args.output)
            
        elif mode == "pickle_to_tar":
            print(f"Converting {args.input} (pickle) → {args.output} (tar)")
            cif_data = load_pickle(args.input)
            pickle_to_tar(cif_data, args.output)
        
        print("Conversion completed successfully!")
        
    except FileNotFoundError:
        print(f"Error: Input file '{args.input}' not found.")
        return 1
    except Exception as e:
        print(f"Error during conversion: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())