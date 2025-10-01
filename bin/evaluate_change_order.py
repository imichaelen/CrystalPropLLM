#!/usr/bin/env python3
"""
Evaluation Script with Changed Block Ordering for CrystalPropLLM.

Evaluate models trained with Operation → Structure → Target block ordering.

Workflow:
1. Generate prompts ending with data_(formula) line
2. Generate CIFs using the trained model
3. Extract properties and create comparison CSV
4. Analyze metrics and create plots
5. Generate evaluation summary report

Usage:
  Basic evaluation with input folder:
    python evaluate_change_order.py --input_folder /path/to/data/ --model /path/to/model/
  
  With custom test file:
    python evaluate_change_order.py --test_file test.pkl.gz --model /path/to/model/ --output_dir results/
  
  Custom generation parameters:
    python evaluate_change_order.py --input_folder /path/to/data/ --model /path/to/model/ --num_gens 5 --temperature 0.8
"""

import os
import argparse
import gzip
import pickle
import tarfile
import csv
import re
import io
import time
import multiprocessing as mp
from datetime import datetime
from contextlib import nullcontext
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Import CrystalPropLLM components
from crystalpropllm import (
    CIFTokenizer,
    GPT,
    GPTConfig,
    array_split,
)

# Regex patterns for prompt extraction - adapted for changed ordering
# For changed order (Operation → Structure → Target), prompts should end with data_(formula)
PATTERN_COMP = re.compile(r"(data_[^\n]*\n)", re.MULTILINE)
PATTERN_COMP_SG = re.compile(r"(data_[^\n]*\n)loop_[\s\S]*?(_symmetry_space_group_name_H-M[^\n]*\n)", re.MULTILINE)
PATTERN_COMP_OPERATION = re.compile(r"(OPERATION_[\s\S]*?)(data_[^\n]*\n)", re.MULTILINE)
PATTERN_COMP_SG_OPERATION = re.compile(r"(OPERATION_[\s\S]*?data_[^\n]*\n)loop_[\s\S]*?(_symmetry_space_group_name_H-M[^\n]*\n)", re.MULTILINE)


def generate_default_paths(input_folder, base_name=None):
    """Generate default paths based on input folder."""
    if base_name is None:
        # Try to infer base name from existing files
        try:
            files = os.listdir(input_folder)
            test_files = [f for f in files if f.endswith('_test.pkl.gz')]
            if test_files:
                # Prefer preprocessed files over regular test files
                preprocessed_files = [f for f in test_files if 'preprocessed' in f]
                if preprocessed_files:
                    base_name = preprocessed_files[0].replace('_test.pkl.gz', '')
                else:
                    base_name = test_files[0].replace('_test.pkl.gz', '')
            else:
                base_name = os.path.basename(os.path.normpath(input_folder))
                if not base_name:
                    base_name = "data"
        except OSError:
            base_name = "data"
    
    test_path = os.path.join(input_folder, f"{base_name}_test.pkl.gz")
    output_dir = os.path.join(input_folder, f"{base_name}_evaluation")
    
    return test_path, output_dir


def auto_detect_vocab_file(input_folder=None, model_dir=None, explicit_vocab_file=None):
    """Auto-detect vocabulary file from various locations."""
    # If explicitly specified, check if it exists
    if explicit_vocab_file:
        if os.path.exists(explicit_vocab_file):
            return explicit_vocab_file
        else:
            print(f"Warning: Specified vocabulary file {explicit_vocab_file} not found, trying auto-detection")
    
    # Try to find voc.txt in various locations
    candidate_paths = []
    
    # 1. Input folder (highest priority)
    if input_folder and os.path.exists(input_folder):
        candidate_paths.append(os.path.join(input_folder, "voc.txt"))
    
    # 2. Model directory
    if model_dir and os.path.exists(model_dir):
        candidate_paths.append(os.path.join(model_dir, "voc.txt"))
    
    # 3. Current working directory
    candidate_paths.append("voc.txt")
    
    # Return the first existing vocabulary file
    for vocab_path in candidate_paths:
        if os.path.exists(vocab_path):
            return vocab_path
    
    return None


def extract_prompt(cif_str, pattern):
    """Extract prompts from CIF string using regex pattern for changed order."""
    match = re.search(pattern, cif_str)
    if match:
        if pattern == PATTERN_COMP_OPERATION:
            # For operation patterns, include operation block + data line
            groups = match.groups()
            prompt = groups[0] + groups[1]
        elif pattern == PATTERN_COMP_SG_OPERATION:
            # For operation + space group pattern, include operation block + structure up to space group
            groups = match.groups()
            prompt = groups[0] + groups[1]
        else:
            # For simple patterns (PATTERN_COMP, PATTERN_COMP_SG), use original logic
            end_index = match.end()
            start_index = match.start()
            prompt = cif_str[start_index:end_index]
        
        # Strip out any leading or trailing spaces from the prompt
        prompt = re.sub(r"^[ \t]+|[ \t]+$", "", prompt, flags=re.MULTILINE)
        # Remove trailing newline to ensure prompt ends exactly with data_(formula)
        prompt = prompt.rstrip()
        return prompt
    else:
        raise Exception(f"Could not extract pattern from CIF: {cif_str[:200]}...")


def generate_prompts(test_file, output_dir, with_spacegroup=False, with_operation=True):
    """Generate prompts from test data and save to tarball."""
    print(f"Generating prompts from {test_file}...")
    
    with gzip.open(test_file, "rb") as f:
        test_data = pickle.load(f)
    
    # Determine which pattern to use based on flags
    if with_operation and with_spacegroup:
        pattern = PATTERN_COMP_SG_OPERATION
    elif with_operation:
        pattern = PATTERN_COMP_OPERATION
    elif with_spacegroup:
        pattern = PATTERN_COMP_SG
    else:
        pattern = PATTERN_COMP
    
    prompts_file = os.path.join(output_dir, "prompts.tar.gz")
    
    with tarfile.open(prompts_file, "w:gz") as tar:
        for cif_id, cif_content in tqdm(test_data, desc="Generating prompts"):
            try:
                prompt = extract_prompt(cif_content, pattern)
                
                prompt_info = tarfile.TarInfo(name=f"{cif_id}.txt")
                prompt_bytes = prompt.encode("utf-8")
                prompt_info.size = len(prompt_bytes)
                tar.addfile(prompt_info, io.BytesIO(prompt_bytes))
            except Exception as e:
                print(f"Warning: Could not generate prompt for {cif_id}: {e}")
                continue
    
    print(f"Prompts saved to {prompts_file}")
    return prompts_file


def progress_listener(queue, n):
    """Progress listener for multiprocessing."""
    pbar = tqdm(total=n, desc="Generating CIFs from prompts")
    while True:
        message = queue.get()
        if message == "kill":
            pbar.close()
            break
        pbar.update(message)


def get_prompts_from_file(prompts_file_path):
    """Load prompts from tarball."""
    prompts_list = []
    with tarfile.open(prompts_file_path, "r:gz") as tar_archive:
        for member in tqdm(tar_archive.getmembers(), desc="Loading prompts"):
            f = tar_archive.extractfile(member)
            if f is not None:
                content = f.read().decode("utf-8")
                filename = os.path.basename(member.name)
                cif_identifier = filename.replace(".txt", "")
                prompts_list.append((cif_identifier, content))
    return prompts_list


def generate_worker(model_dir_path, seed_val, device_str, dtype_str, num_generations, 
                   temperature_val, top_k_val, max_new_tokens_val, chunk_of_prompts_list, queue_obj, vocab_file=None):
    """Worker function for CIF generation."""
    # Initialize torch
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = "cuda" if "cuda" in device_str else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_str]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # Initialize tokenizer
    tokenizer = CIFTokenizer(vocab_file=vocab_file)
    encode = tokenizer.encode
    decode = tokenizer.decode

    # Load model
    print(f"Worker {os.getpid()} initializing model from {model_dir_path} on {device_str}...")
    ckpt_path = os.path.join(model_dir_path, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device_str)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, _v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device_str)
    
    model = torch.compile(model)

    generated_cifs = []
    with torch.no_grad():
        with ctx:
            for cif_id_val, prompt_text in chunk_of_prompts_list:
                tokens, block_ids = tokenizer.tokenize_cif(prompt_text)
                start_ids = encode(tokens)
                x = torch.tensor(start_ids, dtype=torch.long, device=device_str)[None, ...]
                b = torch.tensor(block_ids, dtype=torch.long, device=device_str)[None, ...]
                current_gens = []
                for _ in range(num_generations):
                    y = model.generate(x, max_new_tokens_val, block_ids=b, 
                                     temperature=temperature_val, top_k=top_k_val)
                    output = decode(y[0].tolist())
                    current_gens.append(output)
                generated_cifs.append((cif_id_val, current_gens))
                queue_obj.put(1)
    
    return generated_cifs


def generate_cifs(prompts_file, model_dir, output_dir, device="cuda", dtype="bfloat16", 
                 temperature=1.0, top_k=10, max_new_tokens=3000, num_gens=1, seed=1337, vocab_file=None):
    """Generate CIFs from prompts using the trained model."""
    print(f"Generating CIFs using model from {model_dir}...")
    
    # Check GPU availability
    if device == "cuda":
        gpus_avail = torch.cuda.device_count()
    else:
        gpus_avail = 0
    
    workers = 1 if device == "cpu" else gpus_avail
    if workers == 0:
        workers = 1
        device = "cpu"
    
    # Load prompts
    prompts = get_prompts_from_file(prompts_file)
    
    # Split prompts across workers
    chunks = array_split(prompts, workers)
    manager = mp.Manager()
    queue = manager.Queue()
    
    pool = mp.Pool(workers + 1)
    watcher = pool.apply_async(progress_listener, (queue, len(prompts),))

    jobs = []
    for i in range(workers):
        chunk = chunks[i]
        dev = f"cuda:{i}" if device == "cuda" else device
        worker_seed = seed + i
        job = pool.apply_async(
            generate_worker,
            (model_dir, worker_seed, dev, dtype, num_gens, temperature, top_k, max_new_tokens, chunk, queue, vocab_file)
        )
        jobs.append(job)

    generated = []
    for job in jobs:
        generated.extend(job.get())

    queue.put("kill")
    pool.close()
    pool.join()

    # Save generated CIFs
    generated_file = os.path.join(output_dir, "generated_cifs.tar.gz")
    with tarfile.open(generated_file, "w:gz") as tar_output:
        for cif_id, gens_list in tqdm(generated, desc="Saving generated CIFs"):
            for i, cif_content in enumerate(gens_list):
                cif_file_info = tarfile.TarInfo(name=f"{cif_id}__{i+1}.cif")
                cif_bytes = cif_content.encode("utf-8")
                cif_file_info.size = len(cif_bytes)
                tar_output.addfile(cif_file_info, io.BytesIO(cif_bytes))
    
    print(f"Generated CIFs saved to {generated_file}")
    return generated_file


def extract_properties_from_cif(cif_content):
    """Extract properties from CIF content."""
    # Updated patterns to match the actual CIF format with better regex for property names
    patterns = [
        r'_target_([a-zA-Z0-9_]+)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)',  # target properties (to be predicted)
        r'_opera_([a-zA-Z0-9_]+)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)',   # operation properties (input conditions)
    ]
    
    properties = {}
    for pattern in patterns:
        matches = re.findall(pattern, cif_content, re.MULTILINE)
        for prop_name, prop_value in matches:
            try:
                # Use the full property name including prefix for uniqueness
                if pattern.startswith(r'_target_'):
                    full_name = f"target_{prop_name}"
                elif pattern.startswith(r'_opera_'):
                    full_name = f"opera_{prop_name}"
                
                properties[full_name] = float(prop_value)
            except ValueError:
                continue
    
    return properties


def generate_property_csv(test_file, generated_file, output_dir, aggregation="mean"):
    """Generate CSV comparing true vs predicted properties."""
    print(f"Generating property comparison CSV...")
    
    # Load test data
    print(f"Loading test data from {test_file}...")
    with gzip.open(test_file, "rb") as f:
        test_data = pickle.load(f)
    
    test_dict = {}
    all_properties = set()
    
    for cif_id, cif_content in tqdm(test_data, desc="Processing test data"):
        properties = extract_properties_from_cif(cif_content)
        if properties:
            test_dict[cif_id] = properties
            all_properties.update(properties.keys())
    
    target_props = [p for p in all_properties if p.startswith('target_')]
    opera_props = [p for p in all_properties if p.startswith('opera_')]
    print(f"Loaded {len(test_dict)} test samples")
    print(f"  Target properties (for evaluation): {sorted(target_props)}")
    print(f"  Operation properties (input conditions): {sorted(opera_props)}")
    
    if not target_props:
        print("  Warning: No target properties found - only operation properties (input conditions) present")
        print("  Metrics will only be calculated if target properties are found in generated data")
    
    # Load generated data
    print(f"Loading generated data from {generated_file}...")
    generated_dict = {}
    
    with tarfile.open(generated_file, "r:gz") as tar:
        for member in tqdm(tar.getmembers(), desc="Processing generated data"):
            if member.name.endswith('.cif'):
                f = tar.extractfile(member)
                if f is not None:
                    content = f.read().decode('utf-8')
                    properties = extract_properties_from_cif(content)
                    
                    if properties:
                        # Extract base ID from filename
                        base_name = os.path.basename(member.name)
                        cif_id = re.sub(r'__\d+\.cif$', '', base_name)
                        
                        if cif_id not in generated_dict:
                            generated_dict[cif_id] = []
                        generated_dict[cif_id].append(properties)
                        all_properties.update(properties.keys())
    
    print(f"Loaded generated data for {len(generated_dict)} unique IDs")
    
    # Aggregate predictions
    aggregated_generated = {}
    for cif_id, property_lists in generated_dict.items():
        aggregated_generated[cif_id] = {}
        
        for prop_name in all_properties:
            values = []
            for prop_dict in property_lists:
                if prop_name in prop_dict:
                    values.append(prop_dict[prop_name])
            
            if values:
                if aggregation == 'mean':
                    aggregated_generated[cif_id][prop_name] = np.mean(values)
                elif aggregation == 'median':
                    aggregated_generated[cif_id][prop_name] = np.median(values)
                elif aggregation == 'first':
                    aggregated_generated[cif_id][prop_name] = values[0]
    
    # Create comparison data
    matched_data = []
    all_properties = sorted(all_properties)
    
    for cif_id in test_dict:
        if cif_id in aggregated_generated:
            row = [cif_id]
            for prop_name in all_properties:
                true_value = test_dict[cif_id].get(prop_name, '')
                pred_value = aggregated_generated[cif_id].get(prop_name, '')
                row.extend([true_value, pred_value])
            matched_data.append(row)
    
    print(f"Found {len(matched_data)} matched samples")
    
    # Save to CSV
    csv_file = os.path.join(output_dir, "property_comparison.csv")
    headers = ['id']
    for prop_name in all_properties:
        headers.extend([f'true_{prop_name}', f'pred_{prop_name}'])
    
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        for row in matched_data:
            writer.writerow(row)
    
    print(f"Property comparison saved to {csv_file}")
    return csv_file, all_properties


def analyze_property(csv_file, property_name, output_dir):
    """Analyze a specific property and create plots."""
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Get true and predicted columns for this property
    true_col = f'true_{property_name}'
    pred_col = f'pred_{property_name}'
    
    if true_col not in df.columns or pred_col not in df.columns:
        print(f"Property '{property_name}' not found in CSV file.")
        return None
    
    # Filter out rows where either true or pred value is empty/NaN
    valid_data = df[(df[true_col] != '') & (df[pred_col] != '') & 
                    pd.notna(df[true_col]) & pd.notna(df[pred_col])].copy()
    
    if len(valid_data) == 0:
        print(f"No valid data found for property '{property_name}'.")
        return None
    
    # Convert to numeric
    valid_data[true_col] = pd.to_numeric(valid_data[true_col])
    valid_data[pred_col] = pd.to_numeric(valid_data[pred_col])
    
    true_values = valid_data[true_col].values
    pred_values = valid_data[pred_col].values
    
    # Calculate metrics
    r2 = r2_score(true_values, pred_values)
    mae = mean_absolute_error(true_values, pred_values)
    rmse = np.sqrt(mean_squared_error(true_values, pred_values))
    
    # Calculate additional metrics
    mean_true = np.mean(true_values)
    std_true = np.std(true_values)
    mean_pred = np.mean(pred_values)
    std_pred = np.std(pred_values)
    
    # Pearson correlation
    correlation = np.corrcoef(true_values, pred_values)[0, 1]
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(true_values, pred_values, alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
    
    # Add perfect prediction line
    min_val = min(np.min(true_values), np.min(pred_values))
    max_val = max(np.max(true_values), np.max(pred_values))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2, label='Perfect prediction')
    
    # Add best fit line
    z = np.polyfit(true_values, pred_values, 1)
    p = np.poly1d(z)
    plt.plot([min_val, max_val], p([min_val, max_val]), 'b-', alpha=0.8, linewidth=2, label=f'Best fit (slope={z[0]:.3f})')
    
    # Labels and title
    plt.xlabel(f'True {property_name.replace("_", " ").title()}', fontsize=12)
    plt.ylabel(f'Predicted {property_name.replace("_", " ").title()}', fontsize=12)
    plt.title(f'{property_name.replace("_", " ").title()} Prediction\nR² = {r2:.4f}, MAE = {mae:.4f}', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Make plot square
    plt.axis('equal')
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(output_dir, f'{property_name}_scatter.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create residual plot
    residuals = pred_values - true_values
    plt.figure(figsize=(10, 6))
    plt.scatter(true_values, residuals, alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.8)
    plt.xlabel(f'True {property_name.replace("_", " ").title()}', fontsize=12)
    plt.ylabel('Residuals (Predicted - True)', fontsize=12)
    plt.title(f'{property_name.replace("_", " ").title()} Residuals', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    residual_plot_file = os.path.join(output_dir, f'{property_name}_residuals.png')
    plt.savefig(residual_plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Return metrics
    metrics = {
        'property_name': property_name,
        'n_samples': len(valid_data),
        'r2_score': r2,
        'mae': mae,
        'rmse': rmse,
        'correlation': correlation,
        'mean_true': mean_true,
        'std_true': std_true,
        'mean_pred': mean_pred,
        'std_pred': std_pred,
        'slope': z[0],
        'intercept': z[1],
        'plot_file': plot_file,
        'residual_plot_file': residual_plot_file
    }
    
    return metrics


def create_summary_log(all_metrics, output_dir):
    """Create a comprehensive summary log."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file = os.path.join(output_dir, 'evaluation_summary.log')
    
    with open(log_file, 'w') as f:
        f.write(f"CrystalPropLLM Evaluation Summary (Changed Order)\n")
        f.write(f"Generated on: {timestamp}\n")
        f.write("=" * 80 + "\n")
        f.write("NOTE: Only target properties are evaluated. Operation properties are input conditions.\n")
        f.write("Block ordering: Operation → Structure → Target\n")
        f.write("=" * 80 + "\n\n")
        
        if not all_metrics or all(m is None for m in all_metrics):
            f.write("No target properties found for evaluation.\n")
            f.write("Only operation properties (input conditions) were present in the data.\n")
            return
        
        for metrics in all_metrics:
            if metrics is None:
                continue
                
            f.write(f"Property: {metrics['property_name']}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Number of samples: {metrics['n_samples']}\n")
            f.write(f"R² Score: {metrics['r2_score']:.6f}\n")
            f.write(f"Mean Absolute Error (MAE): {metrics['mae']:.6f}\n")
            f.write(f"Root Mean Square Error (RMSE): {metrics['rmse']:.6f}\n")
            f.write(f"Pearson Correlation: {metrics['correlation']:.6f}\n")
            f.write(f"Best fit line: y = {metrics['slope']:.6f}x + {metrics['intercept']:.6f}\n")
            f.write(f"True values - Mean: {metrics['mean_true']:.6f}, Std: {metrics['std_true']:.6f}\n")
            f.write(f"Predicted values - Mean: {metrics['mean_pred']:.6f}, Std: {metrics['std_pred']:.6f}\n")
            f.write(f"Scatter plot: {os.path.basename(metrics['plot_file'])}\n")
            f.write(f"Residual plot: {os.path.basename(metrics['residual_plot_file'])}\n")
            f.write("\n")
        
        # Summary table
        f.write("Summary Table\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Property':<25} {'N':<8} {'R²':<10} {'MAE':<10} {'RMSE':<10} {'Corr':<10}\n")
        f.write("-" * 80 + "\n")
        
        for metrics in all_metrics:
            if metrics is None:
                continue
            f.write(f"{metrics['property_name']:<25} {metrics['n_samples']:<8} "
                   f"{metrics['r2_score']:<10.4f} {metrics['mae']:<10.4f} "
                   f"{metrics['rmse']:<10.4f} {metrics['correlation']:<10.4f}\n")
    
    print(f"Summary log saved to: {log_file}")


def analyze_all_properties(csv_file, all_properties, output_dir):
    """Analyze all properties and create summary."""
    print("Analyzing properties and creating plots...")
    
    # Filter out operation properties since they are input conditions, not predictions
    target_properties = [prop for prop in all_properties if prop.startswith('target_')]
    opera_properties = [prop for prop in all_properties if prop.startswith('opera_')]
    
    if opera_properties:
        print(f"  Excluding operation properties from metrics (input conditions): {opera_properties}")
    
    if not target_properties:
        print("  Warning: No target properties found for evaluation!")
        return []
    
    print(f"  Evaluating target properties: {target_properties}")
    
    all_metrics = []
    for prop_name in target_properties:
        print(f"  Analyzing {prop_name}...")
        metrics = analyze_property(csv_file, prop_name, output_dir)
        all_metrics.append(metrics)
        
        if metrics:
            print(f"    R² = {metrics['r2_score']:.4f}, MAE = {metrics['mae']:.4f}, RMSE = {metrics['rmse']:.4f}")
    
    # Create summary log
    create_summary_log(all_metrics, output_dir)
    
    return all_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation of CrystalPropLLM on test data with changed block ordering."
    )
    
    # Input options
    parser.add_argument("--input_folder", type=str, default=None,
                        help="Input folder containing test data. Will look for <base_name>_test.pkl.gz")
    parser.add_argument("--base_name", type=str, default=None,
                        help="Base name for files when using --input_folder")
    parser.add_argument("--test_file", type=str, default=None,
                        help="Path to test data file (.pkl.gz). Required if --input_folder is not used")
    
    # Model and output
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the trained model directory")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for results. Auto-generated if using --input_folder")
    parser.add_argument("--vocab_file", type=str, default=None,
                        help="Path to vocabulary file. If not specified, will auto-detect voc.txt in input folder or model directory")
    
    # Generation parameters
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="Device to use for generation")
    parser.add_argument("--dtype", type=str, default="bfloat16", 
                        choices=["float32", "bfloat16", "float16"],
                        help="Data type for generation")
    parser.add_argument("--temperature", type=float, default=0.01,
                        help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Top-k sampling parameter")
    parser.add_argument("--max_new_tokens", type=int, default=3000,
                        help="Maximum new tokens to generate")
    parser.add_argument("--num_gens", type=int, default=1,
                        help="Number of generations per prompt")
    parser.add_argument("--seed", type=int, default=1337,
                        help="Random seed")
    
    # Prompt options - changed from with_property to with_operation
    parser.add_argument("--with_spacegroup", action="store_true",
                        help="Include space group in prompts")
    parser.add_argument("--with_operation", action="store_true", default=True,
                        help="Include operation conditions in prompts")
    
    # Analysis options
    parser.add_argument("--aggregation", type=str, default="mean",
                        choices=["mean", "median", "first"],
                        help="How to aggregate multiple generations")
    parser.add_argument("--skip_analysis", action="store_true",
                        help="Skip property analysis and plot generation")
    
    args = parser.parse_args()
    
    # Determine input and output paths
    if args.input_folder:
        if not os.path.exists(args.input_folder):
            parser.error(f"Input folder does not exist: {args.input_folder}")
        
        test_file, output_dir = generate_default_paths(args.input_folder, args.base_name)
        if args.output_dir:
            output_dir = args.output_dir
    else:
        if not args.test_file:
            parser.error("Either --input_folder or --test_file must be specified")
        if not args.output_dir:
            parser.error("--output_dir is required when using --test_file")
        
        test_file = args.test_file
        output_dir = args.output_dir
    
    # Validate inputs
    if not os.path.exists(test_file):
        parser.error(f"Test file not found: {test_file}")
    if not os.path.exists(args.model):
        parser.error(f"Model directory not found: {args.model}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Auto-detect vocabulary file
    vocab_file = auto_detect_vocab_file(
        input_folder=args.input_folder if args.input_folder else os.path.dirname(test_file),
        model_dir=args.model,
        explicit_vocab_file=args.vocab_file
    )
    
    print(f"Evaluation Configuration (Changed Order):")
    print(f"  Test file: {test_file}")
    print(f"  Model: {args.model}")
    print(f"  Output directory: {output_dir}")
    if vocab_file:
        print(f"  Vocabulary file: {vocab_file} ✓")
    else:
        print(f"  Vocabulary file: Using default tokens only")
    print(f"  Generation parameters: temp={args.temperature}, top_k={args.top_k}, num_gens={args.num_gens}")
    print(f"  Block ordering: Operation → Structure → Target")
    print()
    
    start_time = time.time()
    
    try:
        # Step 1: Generate prompts
        print("Step 1/4: Generating prompts...")
        prompts_file = generate_prompts(
            test_file, output_dir, 
            with_spacegroup=args.with_spacegroup,
            with_operation=args.with_operation
        )
        
        # Step 2: Generate CIFs
        print("\nStep 2/4: Generating CIFs...")
        generated_file = generate_cifs(
            prompts_file, args.model, output_dir,
            device=args.device, dtype=args.dtype,
            temperature=args.temperature, top_k=args.top_k,
            max_new_tokens=args.max_new_tokens, num_gens=args.num_gens,
            seed=args.seed, vocab_file=vocab_file
        )
        
        # Step 3: Generate property comparison CSV
        print("\nStep 3/4: Generating property comparison...")
        csv_file, all_properties = generate_property_csv(
            test_file, generated_file, output_dir,
            aggregation=args.aggregation
        )
        
        # Step 4: Analyze properties
        if not args.skip_analysis:
            print("\nStep 4/4: Analyzing properties...")
            print("Note: Only target properties will be evaluated for prediction accuracy.")
            print("Operation properties are input conditions and will not have metrics calculated.")
            analyze_all_properties(csv_file, all_properties, output_dir)
        else:
            print("\nStep 4/4: Skipped (--skip_analysis)")
        
        elapsed_time = time.time() - start_time
        print(f"\nEvaluation completed successfully in {elapsed_time:.2f} seconds!")
        print(f"Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())