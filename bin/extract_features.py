#!/usr/bin/env python3
"""
Feature Extraction Script for CrystalPropLLM.

Extract composition-based or structure-based features from crystal structures using matminer.

Workflow:
1. Load CIF data from pickle file
2. Parse crystal structures with pymatgen
3. Calculate features using matminer featurizers
4. Save features to JSON file
5. Report success/failure statistics

Usage:
  Calculate default magpie features:
    python extract_features.py data.pkl.gz
  
  Calculate RDF features:
    python extract_features.py data.pkl.gz --feature-type rdf
  
  List available features:
    python extract_features.py --list-features
  
  Custom output file:
    python extract_features.py data.pkl.gz --output my_features.json
"""

import argparse
import gzip
import json
import os
from pathlib import Path
from tqdm import tqdm
import warnings

try:
    import cPickle as pickle
except ImportError:
    import pickle

from pymatgen.io.cif import CifParser
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.structure import RadialDistributionFunction

warnings.filterwarnings("ignore")


def parse_structure_from_cif(cif_str):
    """
    Parse a CIF string and return the structure and composition.
    
    Parameters
    ----------
    cif_str : str
        CIF file content as string
        
    Returns
    -------
    tuple
        (Structure, Composition) from pymatgen
    """
    try:
        struct = CifParser.from_str(cif_str).get_structures(primitive=False)[0]
        return struct, struct.composition
    except Exception as e:
        raise ValueError(f"Failed to parse CIF structure: {e}")



def calculate_magpie_features(composition, feature_subset=None, use_tokenizer_friendly_names=True):
    """
    Calculate composition-based features using ElementProperty with Magpie preset.
    
    Parameters
    ----------
    composition : pymatgen.Composition
        The composition to calculate features for
    feature_subset : list, optional
        List of specific feature names to include. If None, uses default subset.
    use_tokenizer_friendly_names : bool, optional
        If True, converts feature names to use underscores instead of spaces
        to avoid tokenization issues. Default: True
        
    Returns
    -------
    dict
        Dictionary of feature names and their values
    """
    # Default features as specified in the requirements - using correct MagpieData feature names
    default_features = [
        'MagpieData mean AtomicWeight',
        'MagpieData avg_dev AtomicWeight', 
        'MagpieData range AtomicWeight',
        'MagpieData mean CovalentRadius',
        'MagpieData avg_dev CovalentRadius',
        'MagpieData range CovalentRadius', 
        'MagpieData mean Electronegativity',
        'MagpieData avg_dev Electronegativity',
        'MagpieData range Electronegativity',
        'molecular_weight'
    ]
    
    if feature_subset is None:
        feature_subset = default_features
    
    try:
        featurizer = ElementProperty.from_preset("magpie")
        values = featurizer.featurize(composition)
        all_features = dict(zip(featurizer.feature_labels(), values))
        
        # Add molecular weight from composition
        all_features['molecular_weight'] = float(composition.weight)
        
        # Filter to only include requested features
        filtered_features = {}
        for feature_name in feature_subset:
            if feature_name in all_features:
                # Convert feature name to tokenizer-friendly format if requested
                if use_tokenizer_friendly_names:
                    # Replace spaces with underscores and make lowercase for consistency
                    tokenizer_friendly_name = feature_name.replace(' ', '_').replace('MagpieData_', 'magpie_')
                    filtered_features[tokenizer_friendly_name] = all_features[feature_name]
                else:
                    filtered_features[feature_name] = all_features[feature_name]
            else:
                print(f"Warning: Feature '{feature_name}' not found in magpie features")
                
        return filtered_features
        
    except Exception as e:
        raise ValueError(f"Failed to calculate magpie features: {e}")


def calculate_rdf_features(structure):
    """
    Calculate structure-based features using Radial Distribution Function.
    
    Parameters
    ----------
    structure : pymatgen.Structure
        The structure to calculate features for
        
    Returns
    -------
    dict
        Dictionary of RDF feature names and their values
    """
    try:
        featurizer = RadialDistributionFunction()
        values = featurizer.featurize(structure)
        features = dict(zip(featurizer.feature_labels(), values))
        return features
        
    except Exception as e:
        raise ValueError(f"Failed to calculate RDF features: {e}")


def process_cif_data(pkl_gz_path, feature_type='magpie', feature_subset=None, output_path=None, use_tokenizer_friendly_names=True):
    """
    Process CIF data from pkl.gz file and calculate features.
    
    Parameters
    ----------
    pkl_gz_path : str
        Path to the input pkl.gz file containing (cif_id, cif_content) tuples
    feature_type : str
        Type of features to calculate ('magpie' or 'rdf')
    feature_subset : list, optional
        List of specific feature names to include
    output_path : str, optional
        Path for output JSON file. If None, uses 'feature.json' in same directory
        
    Returns
    -------
    dict
        Dictionary with cif_id as keys and features as values
    """
    
    # Set default output path if not provided
    if output_path is None:
        output_dir = Path(pkl_gz_path).parent
        output_path = output_dir / "feature.json"
    
    print(f"Loading CIF data from {pkl_gz_path}...")
    
    # Load the pkl.gz file
    try:
        with gzip.open(pkl_gz_path, "rb") as f:
            cif_data = pickle.load(f)
    except Exception as e:
        raise IOError(f"Failed to load pkl.gz file: {e}")
    
    print(f"Loaded {len(cif_data)} CIF structures")
    
    results = {}
    failed_cifs = []
    
    print(f"Calculating {feature_type} features...")
    
    for cif_id, cif_content in tqdm(cif_data, desc="Processing CIFs"):
        try:
            # Parse the structure
            structure, composition = parse_structure_from_cif(cif_content)
            
            # Calculate features based on type
            if feature_type.lower() == 'magpie':
                features = calculate_magpie_features(composition, feature_subset, use_tokenizer_friendly_names)
            elif feature_type.lower() == 'rdf':
                features = calculate_rdf_features(structure)
            else:
                raise ValueError(f"Unsupported feature type: {feature_type}")
            
            results[cif_id] = features
            
        except Exception as e:
            print(f"Failed to process CIF {cif_id}: {e}")
            failed_cifs.append(cif_id)
            continue
    
    print(f"Successfully processed {len(results)} CIFs")
    if failed_cifs:
        print(f"Failed to process {len(failed_cifs)} CIFs: {failed_cifs[:10]}{'...' if len(failed_cifs) > 10 else ''}")
    
    # Save results to JSON
    print(f"Saving features to {output_path}...")
    try:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Features saved successfully to {output_path}")
    except Exception as e:
        raise IOError(f"Failed to save JSON file: {e}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Extract features from CIF structures in pkl.gz format"
    )
    
    parser.add_argument(
        "input_file",
        type=str,
        nargs='?',
        help="Path to pkl.gz file containing (cif_id, cif_content) tuples"
    )
    
    parser.add_argument(
        "--feature-type",
        type=str,
        choices=['magpie', 'rdf'],
        default='magpie',
        help="Type of features to calculate (default: magpie)"
    )
    
    parser.add_argument(
        "--list-features",
        action="store_true",
        help="List all available magpie features and exit"
    )
    
    parser.add_argument(
        "--features",
        type=str,
        nargs='*',
        default=None,
        help="Specific features to calculate (only for magpie). Default: 'MagpieData mean AtomicWeight', 'MagpieData avg_dev AtomicWeight', 'MagpieData range AtomicWeight', 'MagpieData mean CovalentRadius', 'MagpieData avg_dev CovalentRadius', 'MagpieData range CovalentRadius', 'MagpieData mean Electronegativity', 'MagpieData avg_dev Electronegativity', 'MagpieData range Electronegativity', 'molecular_weight'"
    )
    
    parser.add_argument(
        "--tokenizer-friendly",
        action="store_true",
        default=True,
        help="Convert feature names to use underscores instead of spaces for better tokenization compatibility (default: True)"
    )
    
    parser.add_argument(
        "--original-names",
        action="store_true",
        help="Keep original feature names with spaces (overrides --tokenizer-friendly)"
    )
    
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output JSON file path (default: feature.json in same directory as input)"
    )
    
    args = parser.parse_args()
    
    # Handle --list-features option
    if args.list_features:
        print("Available magpie features:")
        try:
            from matminer.featurizers.composition import ElementProperty
            featurizer = ElementProperty.from_preset("magpie")
            features = featurizer.feature_labels()
            for i, feature in enumerate(features, 1):
                print(f"  {i:3d}. {feature}")
            print(f"\nTotal: {len(features)} features")
            print("\nAdditional features:")
            print("  molecular_weight (calculated from composition)")
        except Exception as e:
            print(f"Error loading magpie features: {e}")
        return 0
    
    # Validate input file is provided for non-list operations
    if not args.input_file:
        parser.error("input_file is required unless using --list-features")
    
    # Validate input file
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")
    
    if not args.input_file.endswith('.pkl.gz'):
        print("Warning: Input file doesn't have .pkl.gz extension")
    
    # Determine whether to use tokenizer-friendly names
    use_tokenizer_friendly = args.tokenizer_friendly and not args.original_names
    
    try:
        results = process_cif_data(
            pkl_gz_path=args.input_file,
            feature_type=args.feature_type,
            feature_subset=args.features,
            output_path=args.output,
            use_tokenizer_friendly_names=use_tokenizer_friendly
        )
        
        print(f"\nFeature extraction completed successfully!")
        print(f"Processed {len(results)} structures")
        
        if results:
            # Show sample of features calculated
            sample_cif = next(iter(results))
            sample_features = results[sample_cif]
            print(f"\nSample features for {sample_cif}:")
            for feature_name, value in list(sample_features.items())[:5]:
                print(f"  {feature_name}: {value}")
            if len(sample_features) > 5:
                print(f"  ... and {len(sample_features) - 5} more features")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
