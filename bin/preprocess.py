#!/usr/bin/env python3
"""
Preprocessing Script for CrystalPropLLM.

Preprocess CIF files by adding features, operation conditions, and target properties.

Workflow:
1. Load CIF data from pickle file
2. Auto-detect and load properties.json and feature.json
3. Process CIFs: standardize, add atomic properties, add blocks
4. Generate vocabulary from property/feature/condition keys
5. Save preprocessed CIFs to pickle file

Usage:
  Basic preprocessing:
    python preprocess.py data.pkl.gz
  
  With custom output:
    python preprocess.py data.pkl.gz --out preprocessed.pkl.gz
  
  Skip features:
    python preprocess.py data.pkl.gz --skip-features
  
  With oxidation states:
    python preprocess.py data.pkl.gz --oxi
"""

import argparse
import gzip
import os
import json
from tqdm import tqdm
import multiprocessing as mp
from queue import Empty
from pymatgen.io.cif import CifParser

from crystalpropllm import (
    semisymmetrize_cif,
    replace_data_formula_with_nonreduced_formula,
    add_atomic_props_block,
    round_numbers,
    extract_formula_units,
)

try:
    import cPickle as pickle
except ImportError:
    import pickle

import warnings
import math
warnings.filterwarnings("ignore")


def _parse_structure(cif_str):
    """Return (Structure, Composition) from a raw CIF string."""
    struct = CifParser.from_str(cif_str).get_structures(primitive=False)[0]
    return struct, struct.composition


def add_feature_block(
    cif_str: str,
    row_data: dict | None,
    feature_order: list[str] | None = None,
    skip_features: bool = False,
) -> str:
    """
    Append a FEATURE_ block using features from row_data.

    Parameters
    ----------
    cif_str : str
        Input CIF string
    row_data : dict | None
        Dictionary containing features data
    feature_order : list[str] | None
        Order of features to include in block
    skip_features : bool
        If True, skip adding feature block
    """
    if skip_features:
        return cif_str.rstrip()

    # ------------------------------------------------------------------ step 1
    # auto‑generate or retrieve features
    # --------------------------------------------------------------------------
    features = (row_data or {}).get("features", {})


    if not features:
        return cif_str.rstrip()  # nothing to add

    # ------------------------------------------------------------------ step 2
    # build FEATURE_ block
    # --------------------------------------------------------------------------
    if feature_order is None:
        feature_order = list(features)

    lines = ["FEATURE_"]
    for key in feature_order:
        if key in features and features[key] is not None:
            lines.append(f"_prop_{key} {features[key]}")

    return cif_str.rstrip() + "\n" + "\n".join(lines) 


def add_operation_block(cif_str, conditions):
    """Add operation condition block after OPERATION_"""
    operation_lines = ["OPERATION_"]
    for key, value in conditions.items():
        if value is not None:
            operation_lines.append(f"_opera_{key} {value}")

    operation_block = "\n".join(operation_lines)
    return f"{cif_str}\n{operation_block}"


def add_target_block(cif_str, targets):
    """Add target properties block after TARGET_"""
    target_lines = ["TARGET_"]
    for key, value in targets.items():
        if value is not None:
            target_lines.append(f"_target_{key} {value}")
    
    target_block = "\n".join(target_lines)
    return f"{cif_str}\n{target_block}"


def progress_listener(queue, n):
    pbar = tqdm(total=n)
    tot = 0
    while True:
        if n == 0:
            break
        message = queue.get()
        tot += message
        pbar.update(message)
        if tot == n:
            break


def augment_cif(progress_queue, task_queue, result_queue, oxi, decimal_places, data, feature_order=None, skip_features=False, ):
    augmented_cifs = []

    while not task_queue.empty():
        try:
            composite_id, original_cif_id, cif_str = task_queue.get_nowait()
        except Empty:
            break

        try:
            formula_units = extract_formula_units(cif_str)
            if formula_units == 0:
                raise ValueError("Formula units cannot be zero")

            cif_str = replace_data_formula_with_nonreduced_formula(cif_str)
            cif_str = semisymmetrize_cif(cif_str)
            cif_str = add_atomic_props_block(cif_str, oxi)
            
            row_data = data.get(composite_id, {})
            
            # Add feature block
            cif_str = add_feature_block(cif_str, row_data, feature_order, skip_features)
            
            # Add operation conditions
            op_conditions = row_data.get('operation_conditions', {})
            if op_conditions:
                cif_str = add_operation_block(cif_str, op_conditions)
            
            # Add target properties
            targets = row_data.get('targets', {})
            if targets:
                cif_str = add_target_block(cif_str, targets)
            
            cif_str = round_numbers(cif_str, decimal_places=decimal_places)
            augmented_cifs.append((composite_id, cif_str))
            
        except (ValueError, KeyError, IndexError, AttributeError) as e:
            print(f"Skipping CIF {composite_id} due to error: {e}")
            pass

        progress_queue.put(1)

    result_queue.put(augmented_cifs)


def auto_detect_json_files(cif_file_path):
    """
    Auto-detect properties.json and feature.json files in the same directory as the CIF file.
    
    Parameters
    ----------
    cif_file_path : str
        Path to the CIF pkl.gz file
        
    Returns
    -------
    tuple
        (properties_json_path, features_json_path) - None if files don't exist
    """
    cif_dir = os.path.dirname(cif_file_path)
    
    properties_json = os.path.join(cif_dir, "properties.json")
    features_json = os.path.join(cif_dir, "feature.json")
    
    properties_path = properties_json if os.path.exists(properties_json) else None
    features_path = features_json if os.path.exists(features_json) else None
    
    return properties_path, features_path


def load_json_data(properties_json_path=None, features_json_path=None):
    """
    Load data from properties.json and feature.json files.
    
    Parameters
    ----------
    properties_json_path : str, optional
        Path to properties.json file
    features_json_path : str, optional
        Path to feature.json file
        
    Returns
    -------
    tuple
        (data, composite_ids, all_property_keys, all_feature_keys, all_condition_keys)
    """
    data = {}
    composite_ids = []
    all_property_keys = set()
    all_feature_keys = set()
    all_condition_keys = set()
    
    # Load properties.json
    if properties_json_path and os.path.exists(properties_json_path):
        print(f"Loading properties from {properties_json_path}...")
        with open(properties_json_path, 'r') as f:
            properties_data = json.load(f)
        
        for entry in properties_data:
            cif_id = entry.get('cif_id')
            if not cif_id:
                continue
                
            # Handle conditions (operation conditions)
            conditions = entry.get('condition', {})
            condition_keys = list(conditions.keys()) if conditions else []
            
            # Handle properties (targets)
            properties = entry.get('properties', {})
            property_keys = list(properties.keys()) if properties else []
            
            # Add condition keys and property keys to separate sets
            all_condition_keys.update(condition_keys)
            all_property_keys.update(property_keys)
            
            # Create composite ID from conditions
            composite_id_parts = [str(cif_id)]
            if conditions:
                for key in sorted(condition_keys):
                    if key in conditions and conditions[key] is not None:
                        composite_id_parts.append(str(conditions[key]))
            
            composite_id = "_".join(composite_id_parts)
            
            if composite_id not in data:
                composite_ids.append((composite_id, cif_id))
                data[composite_id] = {
                    'original_cif_id': cif_id,
                    'features': {},
                    'operation_conditions': conditions,
                    'targets': properties,
                }
    
    # Create a mapping from original_cif_id to composite_ids for efficient lookup
    cif_id_to_composite_ids = {}
    for composite_id, original_cif_id in composite_ids:
        if original_cif_id not in cif_id_to_composite_ids:
            cif_id_to_composite_ids[original_cif_id] = []
        cif_id_to_composite_ids[original_cif_id].append(composite_id)
    
    # Load feature.json and merge with existing data
    if features_json_path and os.path.exists(features_json_path):
        print(f"Loading features from {features_json_path}...")
        with open(features_json_path, 'r') as f:
            features_data = json.load(f)
        
        for cif_id, features in features_data.items():
            all_feature_keys.update(features.keys())
            
            # Use the mapping for efficient lookup instead of linear search
            if cif_id in cif_id_to_composite_ids:
                # Add features to all composite IDs with this cif_id
                for composite_id in cif_id_to_composite_ids[cif_id]:
                    data[composite_id]['features'] = features
            else:
                # If no existing entry, create a new one
                composite_id = cif_id
                composite_ids.append((composite_id, cif_id))
                data[composite_id] = {
                    'original_cif_id': cif_id,
                    'features': features,
                    'operation_conditions': {},
                    'targets': {},
                }
                # Update the mapping
                cif_id_to_composite_ids[cif_id] = [composite_id]
    
    return data, composite_ids, all_property_keys, all_feature_keys, all_condition_keys

def generate_vocabulary_from_json(all_property_keys, all_feature_keys, all_condition_keys, vocab_file):
    """
    Generate vocabulary file from JSON property, feature, and condition keys.
    
    Parameters
    ----------
    all_property_keys : set
        Set of all property keys from properties.json (targets)
    all_feature_keys : set
        Set of all feature keys from feature.json  
    all_condition_keys : set
        Set of all condition keys from properties.json (operation conditions)
    vocab_file : str
        Path to output vocabulary file
        
    Returns
    -------
    set
        Set of all vocabulary tokens
    """
    vocabulary = set()
    
    # Add feature tokens (for FEATURE_ blocks)
    for feature_key in all_feature_keys:
        vocabulary.add(f"_prop_{feature_key}")
    
    # Add condition tokens (for OPERATION_ blocks)
    for condition_key in all_condition_keys:
        vocabulary.add(f"_opera_{condition_key}")
    
    # Add target property tokens (for TARGET_ blocks)
    for prop_key in all_property_keys:
        vocabulary.add(f"_target_{prop_key}")
    
    print(f"Generated vocabulary with {len(vocabulary)} unique tokens")
    print(f"  Feature tokens: {len(all_feature_keys)}")
    print(f"  Condition tokens: {len(all_condition_keys)}")
    print(f"  Property tokens: {len(all_property_keys)}")

    with open(vocab_file, 'w', encoding='utf-8') as f:
        for token in sorted(list(vocabulary)):
            f.write(f"{token}\n")
    
    print(f"Vocabulary saved to {vocab_file}")
    return vocabulary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-process CIF files for property prediction.")
    parser.add_argument("name", type=str,
                        help="Path to the file with the CIFs to be pre-processed (gzipped pickle list of (id, cif) tuples).")
    parser.add_argument("--out", "-o", action="store", default=None,
                        help="Path to the output file for pre-processed CIFs (gzipped pickle dump). If not specified, defaults to same directory as input with '_preprocessed.pkl.gz' suffix.")
    
    # Feature generation options
    parser.add_argument("--feature-generation", type=str, choices=['magpie'], default=None,
                        help="Method for automatic feature generation (e.g., 'magpie').")
    parser.add_argument("--feature-order", type=str, nargs="*", default=None,
                        help="Specify the features to include and their order in the FEATURE_ block.")
    parser.add_argument("--skip-features", action="store_true",
                        help="Skip adding the FEATURE_ block.")

    # General processing options
    parser.add_argument("--oxi", action="store_true",
                        help="Include this flag if the CIFs contain oxidation state information.")
    parser.add_argument("--decimal-places", type=int, default=4,
                        help="Number of decimal places to round floating point numbers to.")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of workers to use for processing.")
    parser.add_argument("--vocab-file", type=str, default=None,
                        help="Path to output vocabulary file for tokenization. If not specified, defaults to 'voc.txt' in same directory as input.")

    args = parser.parse_args()

    # Set default output path if not specified
    if args.out is None:
        input_dir = os.path.dirname(args.name)
        input_basename = os.path.basename(args.name)
        # Remove .pkl.gz extension and add _preprocessed.pkl.gz
        if input_basename.endswith('.pkl.gz'):
            output_basename = input_basename[:-7] + '_preprocessed.pkl.gz'
        else:
            output_basename = input_basename + '_preprocessed.pkl.gz'
        args.out = os.path.join(input_dir, output_basename)
        print(f"Output path not specified, defaulting to: {args.out}")

    # Set default vocab file path if not specified
    if args.vocab_file is None:
        input_dir = os.path.dirname(args.name)
        args.vocab_file = os.path.join(input_dir, 'voc.txt')
        print(f"Vocab file path not specified, defaulting to: {args.vocab_file}")

    # Auto-detect properties.json and feature.json in the same directory
    properties_json_path, features_json_path = auto_detect_json_files(args.name)
    
    # Determine whether to skip features based on availability of feature.json
    skip_features = features_json_path is None
    if skip_features:
        print("No feature.json found - skipping feature blocks")
    else:
        print(f"Found feature.json - will include feature blocks")
    
    if properties_json_path or features_json_path:
        print("Auto-detecting JSON files in CIF directory...")
        if properties_json_path:
            print(f"  Found properties file: {properties_json_path}")
        if features_json_path:
            print(f"  Found features file: {features_json_path}")
            
        data, composite_ids, all_property_keys, all_feature_keys, all_condition_keys = load_json_data(
            properties_json_path, features_json_path
        )
        print(f"Loaded {len(composite_ids)} data entries from JSON files")
    else:
        print("No JSON files found in CIF directory, processing CIFs without additional data")
        data, composite_ids = {}, []
        all_property_keys, all_feature_keys, all_condition_keys = set(), set(), set()

    # Generate vocabulary from detected keys
    if args.vocab_file and (all_property_keys or all_feature_keys or all_condition_keys):
        print("Generating vocabulary file from JSON data...")
        generate_vocabulary_from_json(all_property_keys, all_feature_keys, all_condition_keys, args.vocab_file)

    print(f"Loading CIF data from {args.name}...")
    with gzip.open(args.name, "rb") as f:
        cifs = pickle.load(f)
    
    cif_id_to_str = {cif_id: cif_str for cif_id, cif_str in cifs}
    print(f"Loaded {len(cifs)} CIF structures")

    if not data:
        composite_ids = [(cif_id, cif_id) for cif_id, _ in cifs]

    manager = mp.Manager()
    progress_queue = manager.Queue()
    task_queue = manager.Queue()
    result_queue = manager.Queue()

    valid_tasks_count = 0
    if data:
        for composite_id, original_cif_id in composite_ids:
            if original_cif_id in cif_id_to_str:
                cif_str = cif_id_to_str[original_cif_id]
                task_queue.put((composite_id, original_cif_id, cif_str))
                valid_tasks_count += 1
            else:
                print(f"Warning: CIF ID {original_cif_id} not found in CIF data")
    else:
        for cif_id, cif_str in cifs:
            task_queue.put((cif_id, cif_id, cif_str))
            valid_tasks_count += 1


    print(f"Processing {valid_tasks_count} valid CIFs")

    watcher = mp.Process(target=progress_listener, args=(progress_queue, valid_tasks_count,))

    processes = [mp.Process(target=augment_cif, args=(progress_queue, task_queue, result_queue, args.oxi, args.decimal_places, data, args.feature_order, skip_features))
                 for _ in range(args.workers)]
    processes.append(watcher)

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    modified_cifs = []
    while not result_queue.empty():
        modified_cifs.extend(result_queue.get())

    print(f"Successfully processed {len(modified_cifs)} CIFs")

    print(f"Saving data to {args.out}...")
    with gzip.open(args.out, "wb") as f:
        pickle.dump(modified_cifs, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Processing complete!")
