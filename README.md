# CrystalPropLLM

A Large Language Model framework for predicting crystal properties from crystallographic structures.

> **Note**: This is a research project currently under development. Full documentation and detailed methodology will be available upon publication.

## Overview

CrystalPropLLM is a deep learning framework that applies transformer-based models to predict material properties from crystallographic information.

## Features

- Tokenization of crystal structures from CIF files
- GPT-based model training for property prediction
- Support for composition descriptors and operational conditions
- Comprehensive evaluation metrics

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Conda or Miniconda

### Setup

```bash
# Create conda environment
conda env create -f environment.yml
conda activate cryprop

# Install package
pip install -e .
```

## Quick Start

### Basic Workflow

1. **Prepare your data**:

```bash
python bin/prepare_custom.py --input_dir /path/to/cifs --output_tar_gz cifs.tar.gz
```

2. **Preprocess**:

```bash
python bin/preprocess.py --name cifs.tar.gz --vocab_file voc.txt
```

3. **Tokenize**:

```bash
python bin/tokenizer.py --input_folder preprocessed_data/ --vocab_file voc.txt
```

4. **Train**:

```bash
python bin/train.py --data_dir tokenized_data/ --out_dir out/model/
```

5. **Evaluate**:

```bash
python bin/evaluate.py --input_folder preprocessed_data/ --model out/model/
```

## Project Structure

```
.
├── bin/                    # Main executable scripts
├── crystalpropllm/        # Core library
├── benchmark/             # Baseline models (Random Forest, GNN)
├── environment.yml        # Conda environment specification
└── pyproject.toml        # Package configuration
```

## Citation

This work is currently under review. Citation information will be provided upon publication.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or collaboration inquiries, please open an issue on GitHub.

---

**Status**: Under active development. More comprehensive documentation coming soon.
