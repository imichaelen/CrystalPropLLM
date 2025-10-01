#!/usr/bin/env python3
"""
Data Splitting Script for CrystalPropLLM.

Splits CIF data into training, validation, and test sets.

Workflow:
1. Load CIF data from pickle file
2. Split into train/val/test sets using sklearn
3. Save each set to separate pickle files

Usage:
  Basic split with default parameters:
    python split.py data.pkl.gz
  
  Custom split ratios:
    python split.py data.pkl.gz --validation_size 0.15 --test_size 0.05
  
  Specify output files:
    python split.py data.pkl.gz --train_out train.pkl.gz --val_out val.pkl.gz --test_out test.pkl.gz
"""

import gzip
import pickle
import argparse
import os
from sklearn.model_selection import train_test_split


def generate_default_output_paths(input_path):
    """Generate default output paths based on input filename."""
    # Get directory and base filename without extension
    dir_path = os.path.dirname(input_path)
    base_name = os.path.basename(input_path)
    
    # Remove common extensions to get the base name
    if base_name.endswith('.pkl.gz'):
        base_name = base_name[:-7]  # Remove .pkl.gz
    elif base_name.endswith('.pickle.gz'):
        base_name = base_name[:-10]  # Remove .pickle.gz
    elif base_name.endswith('.gz'):
        base_name = base_name[:-3]  # Remove .gz
        if base_name.endswith('.pkl') or base_name.endswith('.pickle'):
            base_name = base_name.rsplit('.', 1)[0]  # Remove .pkl or .pickle
    
    # Generate default paths
    train_path = os.path.join(dir_path, f"{base_name}_train.pkl.gz")
    val_path = os.path.join(dir_path, f"{base_name}_val.pkl.gz")
    test_path = os.path.join(dir_path, f"{base_name}_test.pkl.gz")
    
    return train_path, val_path, test_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split CIF data into train, validation, and test sets.")
    parser.add_argument("name", type=str,
                        help="Path to the file with the CIFs to be split. It is expected that the file "
                             "contains the gzipped contents of a pickled Python list of tuples, of (id, cif) "
                             "pairs.")
    parser.add_argument("--train_out", type=str, default=None,
                        help="Path to the file where the training set CIFs will be stored. "
                             "If not provided, defaults to '<input_name>_train.pkl.gz' in the same directory. "
                             "The file will contain the gzipped contents of a pickle dump.")
    parser.add_argument("--val_out", type=str, default=None,
                        help="Path to the file where the validation set CIFs will be stored. "
                             "If not provided, defaults to '<input_name>_val.pkl.gz' in the same directory. "
                             "The file will contain the gzipped contents of a pickle dump.")
    parser.add_argument("--test_out", type=str, default=None,
                        help="Path to the file where the test set CIFs will be stored. "
                             "If not provided, defaults to '<input_name>_test.pkl.gz' in the same directory. "
                             "The file will contain the gzipped contents of a pickle dump.")
    parser.add_argument("--random_state", type=int, default=1337,
                        help="Random state for train-test split.")
    parser.add_argument("--validation_size", type=float, default=0.10,
                        help="Size of the validation set as a fraction.")
    parser.add_argument("--test_size", type=float, default=0.02,
                        help="Size of the test set as a fraction.")
    args = parser.parse_args()

    cifs_fname = args.name
    
    # Generate default output paths if not provided
    default_train, default_val, default_test = generate_default_output_paths(cifs_fname)
    
    train_fname = args.train_out if args.train_out else default_train
    val_fname = args.val_out if args.val_out else default_val
    test_fname = args.test_out if args.test_out else default_test
    
    random_state = args.random_state
    validation_size = args.validation_size
    test_size = args.test_size

    print(f"Input file: {cifs_fname}")
    print("Output files:")
    print(f"  Train: {train_fname}")
    print(f"  Validation: {val_fname}")
    print(f"  Test: {test_fname}")
    print()

    print(f"loading data from {cifs_fname}...")
    with gzip.open(cifs_fname, "rb") as f:
        cifs = pickle.load(f)

    print("splitting dataset...")

    cifs_train, cifs_test = train_test_split(cifs, test_size=test_size,
                                             shuffle=True, random_state=random_state)

    cifs_train, cifs_val = train_test_split(cifs_train, test_size=validation_size,
                                            shuffle=True, random_state=random_state)

    print(f"number of CIFs in train set: {len(cifs_train):,}")
    print(f"number of CIFs in validation set: {len(cifs_val):,}")
    print(f"number of CIFs in test set: {len(cifs_test):,}")

    print("writing train set...")
    with gzip.open(train_fname, "wb") as f:
        pickle.dump(cifs_train, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("writing validation set...")
    with gzip.open(val_fname, "wb") as f:
        pickle.dump(cifs_val, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("writing test set...")
    with gzip.open(test_fname, "wb") as f:
        pickle.dump(cifs_test, f, protocol=pickle.HIGHEST_PROTOCOL)
