#!/usr/bin/env python3
"""
Simplified prediction script for CrystalPropLLM.

Workflow:
1. Load CIF files from tar.gz archive
2. Optionally load features.json and operation conditions from properties.json
3. Preprocess CIFs exactly like training: features + conditions -> CIFs
4. Generate prompts from preprocessed CIFs
5. Generate predictions using trained model
6. Extract and save results to CSV

Usage:
  Basic prediction:
    python predict.py --input_cifs cifs.tar.gz --model out/model_dir
  
  With features and operational conditions:
    python predict.py --input_cifs cifs.tar.gz --features_json features.json --properties_json properties.json --model out/model_dir
  
  With custom output directory and generation parameters:
    python predict.py --input_cifs cifs.tar.gz --model out/model_dir --output_dir results --num_gens 5 --temperature 0.2
"""

import os
import argparse
import gzip
import pickle
import tarfile
import json
import csv
import re
import io
import time
import multiprocessing as mp
from datetime import datetime
from contextlib import nullcontext
from tqdm import tqdm

import numpy as np
import torch

from crystalpropllm import (
    CIFTokenizer,
    GPT,
    GPTConfig,
    array_split,
    semisymmetrize_cif,
    replace_data_formula_with_nonreduced_formula,
    add_atomic_props_block,
    round_numbers,
    extract_formula_units,
)


def auto_detect_vocab_file(input_cifs=None, model_dir=None, explicit_vocab_file=None):
    """Auto-detect vocabulary file from various locations."""
    if explicit_vocab_file and os.path.exists(explicit_vocab_file):
        return explicit_vocab_file
    
    # Try to find voc.txt in various locations
    candidate_paths = []
    if input_cifs:
        candidate_paths.append(os.path.join(os.path.dirname(input_cifs), "voc.txt"))
    if model_dir:
        candidate_paths.append(os.path.join(model_dir, "voc.txt"))
    candidate_paths.append("voc.txt")
    
    for vocab_path in candidate_paths:
        if os.path.exists(vocab_path):
            return vocab_path
    return None


def load_cifs_from_tarball(tarball_path):
    """Load CIF files from tar.gz archive."""
    print(f"Loading CIF files from {tarball_path}...")
    cifs = []
    
    with tarfile.open(tarball_path, "r:gz") as tar:
        for member in tqdm(tar.getmembers(), desc="Loading CIFs"):
            if member.name.endswith('.cif'):
                f = tar.extractfile(member)
                if f is not None:
                    content = f.read().decode('utf-8')
                    cif_id = os.path.splitext(os.path.basename(member.name))[0]
                    cifs.append((cif_id, content))
    
    print(f"Loaded {len(cifs)} CIF files")
    return cifs


def load_optional_data(properties_json_path=None, features_json_path=None):
    """Load optional features and operational conditions."""
    features_data = {}
    properties_data = {}
    
    # Load features.json
    if features_json_path and os.path.exists(features_json_path):
        print(f"Loading features from {features_json_path}...")
        with open(features_json_path, 'r') as f:
            features_data = json.load(f)
        print(f"  Loaded features for {len(features_data)} entries")
    
    # Load properties.json (operational conditions)
    if properties_json_path and os.path.exists(properties_json_path):
        print(f"Loading properties from {properties_json_path}...")
        with open(properties_json_path, 'r') as f:
            properties_list = json.load(f)
        
        # Convert list to dict for easier lookup
        for entry in properties_list:
            cif_id = entry.get('cif_id')
            if cif_id:
                properties_data[cif_id] = entry.get('condition', {})
        print(f"  Loaded properties for {len(properties_data)} entries")
    
    return features_data, properties_data


def add_feature_block(cif_str, features):
    """Add FEATURE_ block to CIF."""
    if not features:
        return cif_str
    
    lines = ["FEATURE_"]
    for key, value in features.items():
        if value is not None:
            lines.append(f"_prop_{key} {value}")
    
    return cif_str.rstrip() + "\n" + "\n".join(lines)


def add_operation_block(cif_str, conditions):
    """Add OPERATION_ block to CIF."""
    if not conditions:
        return cif_str
    
    lines = ["OPERATION_"]
    for key, value in conditions.items():
        if value is not None:
            lines.append(f"_opera_{key} {value}")
    
    return cif_str.rstrip() + "\n" + "\n".join(lines)


def preprocess_cifs(cifs, features_data, properties_data, output_dir, oxi=False, decimal_places=4, skip_features=False):
    """Preprocess CIFs exactly like training phase."""
    print("Preprocessing CIF files...")
    
    processed_data = []
    
    for cif_id, cif_str in tqdm(cifs, desc="Processing CIFs"):
        try:
            # Basic CIF processing
            formula_units = extract_formula_units(cif_str)
            if formula_units == 0:
                raise ValueError("Formula units cannot be zero")

            cif_str = replace_data_formula_with_nonreduced_formula(cif_str)
            cif_str = semisymmetrize_cif(cif_str)
            cif_str = add_atomic_props_block(cif_str, oxi)
            
            # Add features if available and not skipped
            if not skip_features and cif_id in features_data:
                cif_str = add_feature_block(cif_str, features_data[cif_id])
            
            # Add operational conditions if available
            if cif_id in properties_data:
                conditions = properties_data[cif_id]
                cif_str = add_operation_block(cif_str, conditions)
                
                # Create composite ID with conditions for uniqueness
                condition_parts = [str(conditions.get(key, '')) for key in sorted(conditions.keys())]
                composite_id = f"{cif_id}_{'_'.join(condition_parts)}"
            else:
                composite_id = cif_id
            
            cif_str = round_numbers(cif_str, decimal_places=decimal_places)
            processed_data.append((composite_id, cif_str))
            
        except Exception as e:
            print(f"Skipping CIF {cif_id} due to error: {e}")
            continue
    
    # Save preprocessed data
    preprocessed_file = os.path.join(output_dir, "preprocessed_cifs.pkl.gz")
    with gzip.open(preprocessed_file, "wb") as f:
        pickle.dump(processed_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Preprocessed {len(processed_data)} CIFs and saved to {preprocessed_file}")
    return preprocessed_file


def generate_prompts(preprocessed_file, output_dir):
    """Generate prompts from preprocessed CIFs."""
    print(f"Generating prompts from {preprocessed_file}...")
    
    with gzip.open(preprocessed_file, "rb") as f:
        preprocessed_data = pickle.load(f)
    
    # Simple pattern to extract everything from data_ line onwards
    pattern = re.compile(r"(data_.*)", re.DOTALL)
    
    prompts_file = os.path.join(output_dir, "prompts.tar.gz")
    
    with tarfile.open(prompts_file, "w:gz") as tar:
        for composite_id, cif_content in tqdm(preprocessed_data, desc="Generating prompts"):
            try:
                match = pattern.search(cif_content)
                if match:
                    prompt = match.group(1).rstrip()
                    # Add TARGET_ block to indicate where prediction should start
                    prompt += "\nTARGET_\n"
                    
                    prompt_info = tarfile.TarInfo(name=f"{composite_id}.txt")
                    prompt_bytes = prompt.encode("utf-8")
                    prompt_info.size = len(prompt_bytes)
                    tar.addfile(prompt_info, io.BytesIO(prompt_bytes))
                else:
                    print(f"Warning: Could not extract prompt for {composite_id}")
            except Exception as e:
                print(f"Warning: Could not generate prompt for {composite_id}: {e}")
    
    print(f"Prompts saved to {prompts_file}")
    return prompts_file


def load_prompts(prompts_file):
    """Load prompts from tarball."""
    prompts = []
    with tarfile.open(prompts_file, "r:gz") as tar:
        for member in tqdm(tar.getmembers(), desc="Loading prompts"):
            f = tar.extractfile(member)
            if f is not None:
                content = f.read().decode("utf-8")
                composite_id = os.path.basename(member.name).replace(".txt", "")
                prompts.append((composite_id, content))
    return prompts


def progress_listener(queue, n):
    """Progress listener for multiprocessing."""
    pbar = tqdm(total=n, desc="Generating predictions")
    while True:
        message = queue.get()
        if message == "kill":
            pbar.close()
            break
        pbar.update(message)


def generate_worker(model_dir, seed, device, dtype, num_gens, temperature, top_k, max_new_tokens, chunk_prompts, queue, vocab_file=None):
    """Worker function for generating predictions."""
    # Initialize torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = "cuda" if "cuda" in device else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # Initialize tokenizer
    tokenizer = CIFTokenizer(vocab_file=vocab_file)
    
    # Load model
    print(f"Worker {os.getpid()} initializing model from {model_dir} on {device}...")
    ckpt_path = os.path.join(model_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    
    # Remove unwanted prefix
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    model = torch.compile(model)

    results = []
    with torch.no_grad():
        with ctx:
            for composite_id, prompt in chunk_prompts:
                tokens, block_ids = tokenizer.tokenize_cif(prompt)
                start_ids = tokenizer.encode(tokens)
                x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
                b = torch.tensor(block_ids, dtype=torch.long, device=device)[None, ...]
                
                generations = []
                for _ in range(num_gens):
                    y = model.generate(x, max_new_tokens, block_ids=b, 
                                     temperature=temperature, top_k=top_k)
                    output = tokenizer.decode(y[0].tolist())
                    generations.append(output)
                
                results.append((composite_id, generations))
                queue.put(1)
    
    return results


def generate_predictions(prompts_file, model_dir, output_dir, device="cuda", dtype="bfloat16", 
                        temperature=0.1, top_k=10, max_new_tokens=2000, num_gens=1, seed=1337, vocab_file=None):
    """Generate predictions using trained model."""
    print(f"Generating predictions using model from {model_dir}...")
    
    # Setup workers
    if device == "cuda":
        workers = torch.cuda.device_count() or 1
    else:
        workers = 1
        device = "cpu"
    
    # Load prompts and split across workers
    prompts = load_prompts(prompts_file)
    chunks = array_split(prompts, workers)
    
    # Run generation
    manager = mp.Manager()
    queue = manager.Queue()
    pool = mp.Pool(workers + 1)
    
    watcher = pool.apply_async(progress_listener, (queue, len(prompts)))
    
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

    # Collect results
    all_results = []
    for job in jobs:
        all_results.extend(job.get())

    queue.put("kill")
    pool.close()
    pool.join()

    # Save generated results
    generated_file = os.path.join(output_dir, "generated_results.tar.gz")
    with tarfile.open(generated_file, "w:gz") as tar:
        for composite_id, generations in tqdm(all_results, desc="Saving results"):
            for i, result in enumerate(generations):
                result_info = tarfile.TarInfo(name=f"{composite_id}__{i+1}.cif")
                result_bytes = result.encode("utf-8")
                result_info.size = len(result_bytes)
                tar.addfile(result_info, io.BytesIO(result_bytes))
    
    print(f"Generated results saved to {generated_file}")
    return generated_file


def extract_predictions(generated_file, output_dir, aggregation="mean"):
    """Extract predictions from generated results and save to CSV."""
    print("Extracting predictions from generated results...")
    
    # Patterns to extract predicted properties
    patterns = [
        r'_opera_([a-zA-Z0-9_]+)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)',   # operational conditions
        r'_prop_([a-zA-Z0-9_]+)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)',    # features
        r'_target_([a-zA-Z0-9_]+)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'   # predicted target properties
    ]
    
    results_dict = {}
    all_properties = set()
    
    with tarfile.open(generated_file, "r:gz") as tar:
        for member in tqdm(tar.getmembers(), desc="Processing results"):
            if member.name.endswith('.cif'):
                f = tar.extractfile(member)
                if f is not None:
                    content = f.read().decode('utf-8')
                    
                    # Extract properties
                    properties = {}
                    for pattern in patterns:
                        matches = re.findall(pattern, content, re.MULTILINE)
                        for prop_name, prop_value in matches:
                            try:
                                if pattern.startswith(r'_opera_'):
                                    full_name = f"opera_{prop_name}"
                                elif pattern.startswith(r'_prop_'):
                                    full_name = f"prop_{prop_name}"
                                else:  # _target_ pattern
                                    full_name = f"target_{prop_name}"
                                
                                properties[full_name] = float(prop_value)
                            except ValueError:
                                continue
                    
                    if properties:
                        # Extract base ID
                        base_name = os.path.basename(member.name)
                        composite_id = re.sub(r'__\d+\.cif$', '', base_name)
                        
                        if composite_id not in results_dict:
                            results_dict[composite_id] = []
                        results_dict[composite_id].append(properties)
                        all_properties.update(properties.keys())
    
    print(f"Extracted predictions for {len(results_dict)} entries")
    print(f"Found properties: {sorted(all_properties)}")
    
    # Aggregate multiple generations
    aggregated = {}
    for composite_id, prop_lists in results_dict.items():
        aggregated[composite_id] = {}
        for prop_name in all_properties:
            values = [props.get(prop_name) for props in prop_lists if prop_name in props]
            if values:
                if aggregation == 'mean':
                    aggregated[composite_id][prop_name] = np.mean(values)
                elif aggregation == 'median':
                    aggregated[composite_id][prop_name] = np.median(values)
                elif aggregation == 'first':
                    aggregated[composite_id][prop_name] = values[0]
    
    # Save to CSV
    csv_file = os.path.join(output_dir, "predictions.csv")
    all_properties = sorted(all_properties)
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        headers = ['id'] + [f'pred_{prop}' for prop in all_properties]
        writer.writerow(headers)
        
        for composite_id in aggregated:
            row = [composite_id]
            for prop in all_properties:
                value = aggregated[composite_id].get(prop, '')
                row.append(value)
            writer.writerow(row)
    
    print(f"Predictions saved to {csv_file}")
    print(f"CSV contains {len(aggregated)} samples with {len(all_properties)} properties")
    return csv_file


def main():
    parser = argparse.ArgumentParser(
        description="Simplified prediction script for CrystalPropLLM"
    )
    
    # Required arguments
    parser.add_argument("--input_cifs", required=True, help="Path to CIF tar.gz file")
    parser.add_argument("--model", required=True, help="Path to trained model directory")
    
    # Optional data
    parser.add_argument("--features_json", help="Path to features.json file")
    parser.add_argument("--properties_json", help="Path to properties.json file (operational conditions)")
    
    # Output and vocab
    parser.add_argument("--output_dir", help="Output directory (auto-generated if not specified)")
    parser.add_argument("--vocab_file", help="Path to vocab file (auto-detected if not specified)")
    
    # Processing options
    parser.add_argument("--oxi", action="store_true", help="Include oxidation states")
    parser.add_argument("--decimal_places", type=int, default=4, help="Decimal places for rounding")
    parser.add_argument("--skip_features", action="store_true", help="Skip FEATURE_ block")
    
    # Generation parameters
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device for generation")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"], help="Data type")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=10, help="Top-k sampling")
    parser.add_argument("--max_new_tokens", type=int, default=2000, help="Max tokens to generate")
    parser.add_argument("--num_gens", type=int, default=1, help="Number of generations per prompt")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument("--aggregation", default="mean", choices=["mean", "median", "first"], help="How to aggregate multiple generations")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.input_cifs):
        parser.error(f"Input CIF file not found: {args.input_cifs}")
    if not os.path.exists(args.model):
        parser.error(f"Model directory not found: {args.model}")
    
    # Set output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"prediction_results_{timestamp}"
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Auto-detect vocab file
    vocab_file = auto_detect_vocab_file(args.input_cifs, args.model, args.vocab_file)
    
    print(f"Prediction Configuration:")
    print(f"  Input CIFs: {args.input_cifs}")
    print(f"  Features JSON: {args.features_json}")
    print(f"  Properties JSON: {args.properties_json}")
    print(f"  Model: {args.model}")
    print(f"  Output: {args.output_dir}")
    print(f"  Vocab file: {vocab_file or 'default tokens only'}")
    print(f"  Generation: temp={args.temperature}, gens={args.num_gens}")
    print()
    
    start_time = time.time()
    
    try:
        # Step 1: Load CIF files
        print("Step 1/5: Loading CIF files...")
        cifs = load_cifs_from_tarball(args.input_cifs)
        
        # Step 2: Load optional data
        print("Step 2/5: Loading optional data...")
        features_data, properties_data = load_optional_data(args.properties_json, args.features_json)
        
        # Step 3: Preprocess CIFs
        print("Step 3/5: Preprocessing CIFs...")
        preprocessed_file = preprocess_cifs(
            cifs, features_data, properties_data, args.output_dir,
            oxi=args.oxi, decimal_places=args.decimal_places, skip_features=args.skip_features
        )
        
        # Step 4: Generate prompts
        print("Step 4/5: Generating prompts...")
        prompts_file = generate_prompts(preprocessed_file, args.output_dir)
        
        # Step 5: Generate predictions
        print("Step 5/6: Generating predictions...")
        generated_file = generate_predictions(
            prompts_file, args.model, args.output_dir,
            device=args.device, dtype=args.dtype, temperature=args.temperature,
            top_k=args.top_k, max_new_tokens=args.max_new_tokens,
            num_gens=args.num_gens, seed=args.seed, vocab_file=vocab_file
        )
        
        # Step 6: Extract predictions
        print("Step 6/6: Extracting predictions...")
        csv_file = extract_predictions(generated_file, args.output_dir, args.aggregation)
        
        elapsed = time.time() - start_time
        print(f"\nPrediction completed in {elapsed:.2f} seconds!")
        print(f"Results: {args.output_dir}")
        print(f"CSV: {csv_file}")
        
    except Exception as e:
        print(f"\nError: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())