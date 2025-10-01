#!/usr/bin/env python3
"""
Tokenization Script for CrystalPropLLM.

Tokenize preprocessed CIF files for model training.

Workflow:
1. Load preprocessed train/val/test CIF data
2. Tokenize CIFs in parallel using multiple workers
3. Encode tokens to integer IDs
4. Save tokenized data as binary files (.bin)
5. Create tar.gz archive with metadata
6. Generate log of unknown tokens

Usage:
  Using input folder (recommended):
    python tokenizer.py --input_folder /path/to/data/
  
  With custom base name:
    python tokenizer.py --input_folder /path/to/data/ --base_name dataset
  
  Using explicit file paths (legacy):
    python tokenizer.py --train_fname train.pkl.gz --val_fname val.pkl.gz --out_dir output/
"""

import os
import numpy as np
import random
import gzip
import argparse
import tarfile
import multiprocessing as mp
from tqdm import tqdm
try:
    import cPickle as pickle
except ImportError:
    import pickle

from crystalpropllm import (
    CIFTokenizer,
    array_split,
)


def generate_default_paths(input_folder, base_name=None):
    """Generate default paths for train, val, test, vocab, and output based on input folder."""
    if base_name is None:
        # Try to infer base name from existing files in the folder
        try:
            files = os.listdir(input_folder)
            train_files = [f for f in files if f.endswith('_train.pkl.gz')]
            if train_files:
                base_name = train_files[0].replace('_train.pkl.gz', '')
            else:
                # Fallback to folder name
                base_name = os.path.basename(os.path.normpath(input_folder))
                if not base_name:  # Handle case where input_folder is '.' or similar
                    base_name = "data"
        except OSError:
            base_name = "data"
    
    train_path = os.path.join(input_folder, f"{base_name}_train.pkl.gz")
    val_path = os.path.join(input_folder, f"{base_name}_val.pkl.gz")
    test_path = os.path.join(input_folder, f"{base_name}_test.pkl.gz")
    vocab_path = os.path.join(input_folder, "voc.txt")
    output_dir = os.path.join(input_folder, f"{base_name}_tokenized")
    
    return train_path, val_path, test_path, vocab_path, output_dir


def progress_listener(queue, n):
    pbar = tqdm(total=n, desc="tokenizing...")
    while True:
        message = queue.get()
        if message == "kill":
            break
        pbar.update(message)


def tokenize(chunk_of_cifs, queue=None, vocab_file=None):
    tokenizer = CIFTokenizer(vocab_file=vocab_file)
    tokenized = []
    block_ids_list = []
    unk_details = []
    for i, cif in enumerate(tqdm(chunk_of_cifs, disable=queue is not None, desc="tokenizing...")):
        if queue:
            queue.put(1)
        tokens, block_ids = tokenizer.tokenize_cif(cif)
        tokenized.append(tokens)
        block_ids_list.append(block_ids)
        
        # Check for <unk> tokens and record details
        unk_count = tokens.count("<unk>")
        if unk_count > 0:
            unk_details.append({
                'chunk_index': i,
                'unk_count': unk_count,
                'total_tokens': len(tokens),
                'unk_percentage': (unk_count / len(tokens)) * 100,
                'cif_preview': cif
            })
    
    return tokenized, block_ids_list, unk_details


def preprocess(cifs_raw):
    cifs = []
    cif_ids = []
    for cif_id, cif in tqdm(cifs_raw, desc="preparing files..."):
        # filter out some lines in the CIF
        lines = cif.split('\n')
        cif_lines = []
        for line in lines:
            line = line.strip()
            if len(line) > 0 and not line.startswith("#") and "pymatgen" not in line:
                cif_lines.append(line)
        cif_lines.append("\n")
        cifs.append("\n".join(cif_lines))
        cif_ids.append(cif_id)
    return cifs, cif_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tokenize CIF files for CrystalPropLLM training."
    )
    
    # Input folder mode (new default approach)
    parser.add_argument("--input_folder", type=str, default=None,
                        help="Input folder containing train/val/test files and voc.txt. "
                             "Expected files: <base_name>_train.pkl.gz, <base_name>_val.pkl.gz, "
                             "<base_name>_test.pkl.gz, voc.txt. Output will be saved to "
                             "<base_name>_tokenized/ subfolder.")
    parser.add_argument("--base_name", type=str, default=None,
                        help="Base name for files when using --input_folder. If not provided, "
                             "will be auto-detected from existing _train.pkl.gz files or use folder name.")
    
    # Legacy explicit file mode
    parser.add_argument("--train_fname", type=str, default=None,
                        help="Path to the file with the training set CIFs to be tokenized. "
                             "Required if --input_folder is not used.")
    parser.add_argument("--val_fname", type=str, default="",
                        help="Path to the file with the validation set CIFs to be tokenized.")
    parser.add_argument("--test_fname", type=str, default="",
                        help="Path to the file with the test set CIFs to be tokenized.")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory to store processed files. "
                             "Required if --input_folder is not used.")
    
    # Common options
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of workers to use for processing.")
    parser.add_argument("--vocab_file", type=str, default=None,
                        help="Path to vocabulary file. If not specified and using --input_folder, "
                             "will look for voc.txt in the input folder.")
    
    args = parser.parse_args()

    # Determine input mode and set up paths
    if args.input_folder:
        if not os.path.exists(args.input_folder):
            parser.error(f"Input folder does not exist: {args.input_folder}")
        
        train_fname, val_fname, test_fname, vocab_file, out_dir = generate_default_paths(
            args.input_folder, args.base_name
        )
        
        # Override vocab_file if explicitly specified
        if args.vocab_file:
            vocab_file = args.vocab_file
        
        print(f"Input folder mode:")
        print(f"  Folder: {args.input_folder}")
        print(f"  Train file: {train_fname}")
        print(f"  Val file: {val_fname}")
        print(f"  Test file: {test_fname}")
        print(f"  Vocab file: {vocab_file}")
        print(f"  Output dir: {out_dir}")
        print()
        
    else:
        # Legacy explicit file mode
        if not args.train_fname:
            parser.error("Either --input_folder or --train_fname must be specified")
        if not args.out_dir:
            parser.error("--out_dir is required when using explicit file mode")
        
        train_fname = args.train_fname
        val_fname = args.val_fname
        test_fname = args.test_fname
        out_dir = args.out_dir
        vocab_file = args.vocab_file

    workers = args.workers
    
    # Check which files exist
    has_train = os.path.exists(train_fname)
    has_val = os.path.exists(val_fname) if val_fname else False
    has_test = os.path.exists(test_fname) if test_fname else False
    
    if not has_train:
        parser.error(f"Training file not found: {train_fname}")
    
    print(f"Files found:")
    print(f"  Train: {'✓' if has_train else '✗'} {train_fname}")
    print(f"  Val: {'✓' if has_val else '✗'} {val_fname if val_fname else 'Not specified'}")
    print(f"  Test: {'✓' if has_test else '✗'} {test_fname if test_fname else 'Not specified'}")
    
    # Auto-detect vocabulary file if not specified
    if vocab_file is None:
        if args.input_folder:
            auto_vocab_file = os.path.join(args.input_folder, "voc.txt")
        else:
            train_dir = os.path.dirname(train_fname)
            auto_vocab_file = os.path.join(train_dir, "voc.txt")
        
        if os.path.exists(auto_vocab_file):
            vocab_file = auto_vocab_file
            print(f"  Vocab: ✓ {vocab_file} (auto-detected)")
        else:
            print(f"  Vocab: ✗ No vocabulary file found, using default tokens only")
    elif vocab_file and not os.path.exists(vocab_file):
        print(f"  Vocab: ✗ Specified vocabulary file {vocab_file} not found, using default tokens only")
        vocab_file = None
    else:
        print(f"  Vocab: ✓ {vocab_file}")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f"Created output directory: {out_dir}")

    print(f"\nLoading data from {train_fname}...")
    with gzip.open(train_fname, "rb") as f:
        cifs_raw_train = pickle.load(f)

    cifs_raw_val = None
    cifs_raw_test = None
    
    if has_val:
        print(f"Loading data from {val_fname}...")
        with gzip.open(val_fname, "rb") as f:
            cifs_raw_val = pickle.load(f)
    
    if has_test:
        print(f"Loading data from {test_fname}...")
        with gzip.open(test_fname, "rb") as f:
            cifs_raw_test = pickle.load(f)

    # shuffle the order of the train CIFs
    random.shuffle(cifs_raw_train)

    cifs_train, cif_ids_train = preprocess(cifs_raw_train)
    cifs_val, cif_ids_val = None, None
    cifs_test, cif_ids_test = None, None
    
    if has_val:
        cifs_val, cif_ids_val = preprocess(cifs_raw_val)
    
    if has_test:
        cifs_test, cif_ids_test = preprocess(cifs_raw_test)

    # tokenize the train CIFs in parallel
    chunks = array_split(cifs_train, workers)
    chunk_ids = array_split(cif_ids_train, workers)
    manager = mp.Manager()
    queue = manager.Queue()
    pool = mp.Pool(workers + 1)  # add an extra worker for the watcher
    watcher = pool.apply_async(progress_listener, (queue, len(cifs_train),))

    jobs = []
    for i in range(workers):
        chunk = chunks[i]
        job = pool.apply_async(tokenize, (chunk, queue, vocab_file))
        jobs.append(job)

    tokenized_cifs_train = []
    block_ids_train_list = []
    all_unk_details_train = []
    for i, job in enumerate(jobs):
        tokenized_chunk, block_ids_chunk, unk_details_chunk = job.get()
        tokenized_cifs_train.extend(tokenized_chunk)
        block_ids_train_list.extend(block_ids_chunk)
        
        # Add the original CIF IDs to the unk details
        chunk_cif_ids = chunk_ids[i]
        for detail in unk_details_chunk:
            detail['cif_id'] = chunk_cif_ids[detail['chunk_index']]
            detail['global_index'] = sum(len(chunks[j]) for j in range(i)) + detail['chunk_index']
        all_unk_details_train.extend(unk_details_chunk)

    queue.put("kill")
    pool.close()
    pool.join()

    lens = [len(t) for t in tokenized_cifs_train]
    unk_counts = [t.count("<unk>") for t in tokenized_cifs_train]
    print(f"train min tokenized length: {np.min(lens):,}")
    print(f"train max tokenized length: {np.max(lens):,}")
    print(f"train mean tokenized length: {np.mean(lens):.2f} +/- {np.std(lens):.2f}")
    print(f"train total unk counts: {np.sum(unk_counts)}")

    if has_val:
        # tokenize the validation CIFs
        tokenized_cifs_val, block_ids_val_list, all_unk_details_val = tokenize(cifs_val, vocab_file=vocab_file)
        
        # Add CIF IDs to validation unk details
        for i, detail in enumerate(all_unk_details_val):
            detail['cif_id'] = cif_ids_val[detail['chunk_index']]
            detail['global_index'] = detail['chunk_index']

        lens = [len(t) for t in tokenized_cifs_val]
        unk_counts = [t.count("<unk>") for t in tokenized_cifs_val]
        print(f"val min tokenized length: {np.min(lens):,}")
        print(f"val max tokenized length: {np.max(lens):,}")
        print(f"val mean tokenized length: {np.mean(lens):.2f} +/- {np.std(lens):.2f}")
        print(f"val total unk counts: {np.sum(unk_counts)}")

    # Process test set if available
    tokenized_cifs_test = None
    block_ids_test_list = None
    all_unk_details_test = []
    
    if has_test:
        print("Tokenizing test set...")
        tokenized_cifs_test, block_ids_test_list, all_unk_details_test = tokenize(cifs_test, vocab_file=vocab_file)
        
        # Add CIF IDs to test unk details
        for i, detail in enumerate(all_unk_details_test):
            detail['cif_id'] = cif_ids_test[detail['chunk_index']]
            detail['global_index'] = detail['chunk_index']

        lens = [len(t) for t in tokenized_cifs_test]
        unk_counts = [t.count("<unk>") for t in tokenized_cifs_test]
        print(f"test min tokenized length: {np.min(lens):,}")
        print(f"test max tokenized length: {np.max(lens):,}")
        print(f"test mean tokenized length: {np.mean(lens):.2f} +/- {np.std(lens):.2f}")
        print(f"test total unk counts: {np.sum(unk_counts)}")

    # create a single stream of tokens that will be the dataset
    train_data = []
    for t in tqdm(tokenized_cifs_train, desc="concatenating train tokens..."):
        train_data.extend(t)

    train_block_ids = []
    for b in tqdm(block_ids_train_list, desc="concatenating train block_ids..."):
        train_block_ids.extend(b)

    if has_val:
        val_data = []
        for t in tqdm(tokenized_cifs_val, desc="concatenating val tokens..."):
            val_data.extend(t)

        val_block_ids = []
        for b in tqdm(block_ids_val_list, desc="concatenating val block_ids..."):
            val_block_ids.extend(b)

    if has_test:
        test_data = []
        for t in tqdm(tokenized_cifs_test, desc="concatenating test tokens..."):
            test_data.extend(t)

        test_block_ids = []
        for b in tqdm(block_ids_test_list, desc="concatenating test block_ids..."):
            test_block_ids.extend(b)

    print("encoding...")
    tokenizer = CIFTokenizer(vocab_file=vocab_file)
    train_ids = tokenizer.encode(train_data)
    print(f"train has {len(train_ids):,} tokens")
    if has_val:
        val_ids = tokenizer.encode(val_data)
        print(f"val has {len(val_ids):,} tokens")
    if has_test:
        test_ids = tokenizer.encode(test_data)
        print(f"test has {len(test_ids):,} tokens")
    print(f"vocab size: {len(tokenizer.token_to_id)}")

    print("exporting to .bin files...")
    train_ids = np.array(train_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(out_dir, "train.bin"))
    train_block_ids = np.array(train_block_ids, dtype=np.uint16)
    train_block_ids.tofile(os.path.join(out_dir, "train_block_ids.bin"))
    if has_val:
        val_ids = np.array(val_ids, dtype=np.uint16)
        val_ids.tofile(os.path.join(out_dir, "val.bin"))
        val_block_ids = np.array(val_block_ids, dtype=np.uint16)
        val_block_ids.tofile(os.path.join(out_dir, "val_block_ids.bin"))
    if has_test:
        test_ids = np.array(test_ids, dtype=np.uint16)
        test_ids.tofile(os.path.join(out_dir, "test.bin"))
        test_block_ids = np.array(test_block_ids, dtype=np.uint16)
        test_block_ids.tofile(os.path.join(out_dir, "test_block_ids.bin"))

    # save the meta information as well, to help us encode/decode later
    meta = {
        "vocab_size": len(tokenizer.token_to_id),
        "itos": tokenizer.id_to_token,
        "stoi": tokenizer.token_to_id,
    }
    with open(os.path.join(out_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    print("creating tar.gz archive...")
    subdir_name = os.path.basename(os.path.normpath(out_dir))
    tar_gz_filename = os.path.join(out_dir, f"{subdir_name}.tar.gz")
    with tarfile.open(tar_gz_filename, "w:gz") as tar:
        files_to_archive = ["train.bin", "train_block_ids.bin", "meta.pkl"]
        if has_val:
            files_to_archive.extend(["val.bin", "val_block_ids.bin"])
        if has_test:
            files_to_archive.extend(["test.bin", "test_block_ids.bin"])
            
        for filename in files_to_archive:
            file_path = os.path.join(out_dir, filename)
            if os.path.exists(file_path):
                arcname = os.path.join(subdir_name, filename)
                tar.add(file_path, arcname=arcname)

    print(f"tarball created at {tar_gz_filename}")
    
    # Write detailed log file for files with <unk> tokens
    log_filename = os.path.join(out_dir, "unk_tokens_log.txt")
    with open(log_filename, "w") as log_file:
        log_file.write("=" * 80 + "\n")
        log_file.write("TOKENIZATION LOG - FILES WITH <unk> TOKENS\n")
        log_file.write("=" * 80 + "\n\n")
        
        # Training set details
        log_file.write(f"TRAINING SET ANALYSIS\n")
        log_file.write(f"Total files processed: {len(cifs_train)}\n")
        log_file.write(f"Files with <unk> tokens: {len(all_unk_details_train)}\n")
        log_file.write(f"Percentage with <unk>: {(len(all_unk_details_train) / len(cifs_train)) * 100:.2f}%\n")
        log_file.write("-" * 80 + "\n\n")
        
        for detail in all_unk_details_train:
            log_file.write(f"CIF ID: {detail['cif_id']}\n")
            log_file.write(f"Global Index: {detail['global_index']}\n")
            log_file.write(f"<unk> Count: {detail['unk_count']}\n")
            log_file.write(f"Total Tokens: {detail['total_tokens']}\n")
            log_file.write(f"<unk> Percentage: {detail['unk_percentage']:.2f}%\n")
            log_file.write(f"CIF Preview:\n{detail['cif_preview']}\n")
            log_file.write("-" * 40 + "\n\n")
        
        if has_val:
            log_file.write(f"\nVALIDATION SET ANALYSIS\n")
            log_file.write(f"Total files processed: {len(cifs_val)}\n")
            log_file.write(f"Files with <unk> tokens: {len(all_unk_details_val)}\n")
            log_file.write(f"Percentage with <unk>: {(len(all_unk_details_val) / len(cifs_val)) * 100:.2f}%\n")
            log_file.write("-" * 80 + "\n\n")
            
            for detail in all_unk_details_val:
                log_file.write(f"CIF ID: {detail['cif_id']}\n")
                log_file.write(f"Global Index: {detail['global_index']}\n")
                log_file.write(f"<unk> Count: {detail['unk_count']}\n")
                log_file.write(f"Total Tokens: {detail['total_tokens']}\n")
                log_file.write(f"<unk> Percentage: {detail['unk_percentage']:.2f}%\n")
                log_file.write(f"CIF Preview:\n{detail['cif_preview']}\n")
                log_file.write("-" * 40 + "\n\n")
        
        if has_test:
            log_file.write(f"\nTEST SET ANALYSIS\n")
            log_file.write(f"Total files processed: {len(cifs_test)}\n")
            log_file.write(f"Files with <unk> tokens: {len(all_unk_details_test)}\n")
            log_file.write(f"Percentage with <unk>: {(len(all_unk_details_test) / len(cifs_test)) * 100:.2f}%\n")
            log_file.write("-" * 80 + "\n\n")
            
            for detail in all_unk_details_test:
                log_file.write(f"CIF ID: {detail['cif_id']}\n")
                log_file.write(f"Global Index: {detail['global_index']}\n")
                log_file.write(f"<unk> Count: {detail['unk_count']}\n")
                log_file.write(f"Total Tokens: {detail['total_tokens']}\n")
                log_file.write(f"<unk> Percentage: {detail['unk_percentage']:.2f}%\n")
                log_file.write(f"CIF Preview:\n{detail['cif_preview']}\n")
                log_file.write("-" * 40 + "\n\n")
    
    print(f"<unk> tokens log saved to {log_filename}")
    print(f"Training files with <unk>: {len(all_unk_details_train)}/{len(cifs_train)}")
    if has_val:
        print(f"Validation files with <unk>: {len(all_unk_details_val)}/{len(cifs_val)}")
    if has_test:
        print(f"Test files with <unk>: {len(all_unk_details_test)}/{len(cifs_test)}")
