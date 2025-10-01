#!/usr/bin/env python3
"""
CSV to JSON Converter for Crystal Property Data.

Convert CSV files with crystal data to structured JSON format, separating
operational conditions from target properties.

Workflow:
1. Read CSV file with crystal data
2. Identify condition columns (e.g., temperature, pressure)
3. Identify property columns (e.g., band_gap, formation_energy)
4. Convert to JSON format with cif_id, conditions, and properties
5. Save to properties.json file

Usage:
  With conditions:
    python csv_to_json_converter.py test.csv --condition-start 1 --condition-end 2 --property-start 3
  
  Without conditions:
    python csv_to_json_converter.py simple.csv --property-start 1
  
  Custom output file:
    python csv_to_json_converter.py test.csv -cs 1 -ce 2 -ps 3 -o custom_output.json
"""

import argparse
import pandas as pd
import json
import os
from pathlib import Path


def convert_csv_to_json(csv_file, condition_start, condition_end, property_start, output_file=None):
    """
    Convert CSV file to structured JSON format.
    
    Args:
        csv_file (str): Path to input CSV file
        condition_start (int): Starting column index for conditions (0-based, None if no conditions)
        condition_end (int): Ending column index for conditions (inclusive, 0-based, None if no conditions)
        property_start (int): Starting column index for properties (0-based, properties go to end)
        output_file (str, optional): Output JSON file path. Defaults to input_with_conditions.json
    """
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Get column names
    columns = df.columns.tolist()
    
    # Extract column groups
    cif_id_col = columns[0]  # First column is assumed to be cif_id
    
    # Handle case with no conditions
    if condition_start is None or condition_end is None:
        condition_cols = []
        has_conditions = False
    else:
        condition_cols = columns[condition_start:condition_end + 1]
        has_conditions = len(condition_cols) > 0
    
    property_cols = columns[property_start:]
    
    print(f"CIF ID column: {cif_id_col}")
    if has_conditions:
        print(f"Condition columns: {condition_cols}")
    else:
        print("No condition columns specified")
    print(f"Property columns: {property_cols}")
    
    # Convert to JSON structure
    json_data = []
    
    for _, row in df.iterrows():
        # Create entry starting with cif_id
        entry = {
            "cif_id": row[cif_id_col]
        }
        
        # Add condition data only if conditions exist
        if has_conditions:
            condition_data = {}
            for col in condition_cols:
                condition_data[col] = row[col]
            entry["condition"] = condition_data
        
        # Extract property data
        property_data = {}
        for col in property_cols:
            property_data[col] = row[col]
        entry["properties"] = property_data
        
        json_data.append(entry)
    
    # Determine output file name
    if output_file is None:
        input_path = Path(csv_file)
        output_file = input_path.parent / f"properties.json"
    
    # Write JSON file
    with open(output_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"Successfully converted {csv_file} to {output_file}")
    print(f"Generated {len(json_data)} entries")


def main():
    parser = argparse.ArgumentParser(
        description="Convert CSV files to structured JSON format for crystal property data"
    )
    
    parser.add_argument(
        'csv_file',
        help='Input CSV file path'
    )
    
    parser.add_argument(
        '--condition-start', '-cs',
        type=int,
        help='Starting column index for conditions (0-based, first column after cif_id). Omit for no conditions.'
    )
    
    parser.add_argument(
        '--condition-end', '-ce',
        type=int,
        help='Ending column index for conditions (inclusive, 0-based). Omit for no conditions.'
    )
    
    parser.add_argument(
        '--property-start', '-ps',
        type=int,
        required=True,
        help='Starting column index for properties (0-based, properties extend to end of file)'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output JSON file path (default: input_filename_with_conditions.json)'
    )
    
    parser.add_argument(
        '--preview', '-p',
        action='store_true',
        help='Preview column assignments without converting'
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.csv_file):
        print(f"Error: Input file '{args.csv_file}' not found")
        return 1
    
    # Validate column indices
    df_preview = pd.read_csv(args.csv_file)
    num_cols = len(df_preview.columns)
    
    # Check if conditions are specified
    has_conditions = args.condition_start is not None and args.condition_end is not None
    
    if has_conditions:
        if args.condition_start >= num_cols or args.condition_end >= num_cols:
            print(f"Error: Condition column indices out of range. CSV has {num_cols} columns (0-{num_cols-1})")
            return 1
        
        if args.condition_start > args.condition_end:
            print("Error: condition-start must be <= condition-end")
            return 1
        
        if args.condition_end >= args.property_start:
            print("Error: condition-end must be < property-start")
            return 1
    elif args.condition_start is not None or args.condition_end is not None:
        print("Error: Both --condition-start and --condition-end must be specified together, or both omitted")
        return 1
    
    if args.property_start >= num_cols:
        print(f"Error: Property start column index out of range. CSV has {num_cols} columns (0-{num_cols-1})")
        return 1
    
    # Preview mode
    if args.preview:
        columns = df_preview.columns.tolist()
        has_conditions = args.condition_start is not None and args.condition_end is not None
        
        print(f"\nCSV file: {args.csv_file}")
        print(f"Total columns: {num_cols}")
        print(f"Column mapping:")
        print(f"  CIF ID: {columns[0]} (column 0)")
        
        if has_conditions:
            print(f"  Conditions: {columns[args.condition_start:args.condition_end + 1]} (columns {args.condition_start}-{args.condition_end})")
        else:
            print("  Conditions: None")
            
        print(f"  Properties: {columns[args.property_start:]} (columns {args.property_start}-{num_cols-1})")
        print(f"\nSample data (first 3 rows):")
        print(df_preview.head(3).to_string(index=False))
        return 0
    
    # Convert file
    try:
        convert_csv_to_json(
            args.csv_file,
            args.condition_start,
            args.condition_end,
            args.property_start,
            args.output
        )
        return 0
    except Exception as e:
        print(f"Error during conversion: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
