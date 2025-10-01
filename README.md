# CrystalPropLLM

A Large Language Model framework for predicting crystal properties from CIF (Crystallographic Information File) structures. This project uses transformer-based models to generate and predict thermoelectric and other material properties.

## Overview

CrystalPropLLM is a deep learning framework that:

- Tokenizes crystal structures from CIF files
- Trains GPT-based models to predict material properties
- Evaluates predictions against ground truth data
- Supports features like composition descriptors, operational conditions, and target properties

## Project Structure

```
.
├── bin/                          # Main executable scripts
│   ├── train.py                  # Model training script
│   ├── evaluate.py               # Standard evaluation pipeline
│   ├── evaluate_change_order.py  # Evaluation with modified block ordering
│   ├── predict.py                # Prediction on new data
│   ├── preprocess.py             # Data preprocessing
│   ├── preprocess_change_order.py # Preprocessing with modified ordering
│   ├── tokenizer.py              # CIF tokenization
│   ├── extract_features.py       # Feature extraction (Magpie, RDF)
│   ├── csv_to_json_converter.py  # Convert CSV to JSON format
│   └── prepare_custom.py         # Prepare custom CIF datasets
│
├── crystalpropllm/               # Core library
│   └── _metrics.py               # Evaluation metrics (bond length, space group)
│
├── benchmark/                    # Baseline models and data preparation
│   ├── multiGNN/                 # Graph Neural Network baseline
│   │   ├── training.py           # GNN training
│   │   ├── eval_model.py         # GNN evaluation
│   │   └── Data.py               # GNN data loading
│   ├── randomforest/             # Random Forest baseline
│   │   ├── rf.py                 # Main RF implementation
│   │   ├── rf_seebeck.py         # Seebeck coefficient prediction
│   │   ├── rf_sigma.py           # Conductivity prediction
│   │   └── rf_power_factor.py    # Power factor prediction
│   └── prepare_input_file/       # Data preparation utilities
│       ├── combine_data.py       # Combine features and properties
│       └── feature_generation.py # Extract Materials Project features
│
├── environment.yml               # Conda environment specification
├── pyproject.toml               # Python package configuration
└── materials_data_full.json     # Full materials dataset
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA 12.8+ (for GPU support)
- Conda or Miniconda

### Setup Environment

```bash
# Create conda environment
conda env create -f environment.yml
conda activate cryprop

# Install package
pip install -e .
```

## Data Preparation

### 1. Prepare CIF Files

```bash
# From custom CIF directory
python bin/prepare_custom.py --input_dir /path/to/cifs --output_tar_gz cifs.tar.gz
```

### 2. Extract Features (Optional)

```bash
# Extract Magpie composition features
python bin/extract_features.py --input cifs.pkl.gz --feature_type magpie --output feature.json

# Extract RDF structure features
python bin/extract_features.py --input cifs.pkl.gz --feature_type rdf --output feature.json
```

### 3. Convert Properties to JSON

```bash
# Convert CSV with conditions and properties
python bin/csv_to_json_converter.py input.csv \
    --condition-start 1 --condition-end 2 \
    --property-start 3 --output properties.json
```

### 4. Preprocess Data

```bash
# Standard preprocessing (Structure → Feature → Target)
python bin/preprocess.py \
    --name cifs.tar.gz \
    --vocab_file voc.txt \
    --train_ratio 0.8 --val_ratio 0.1

# Alternative ordering (Operation → Structure → Target)
python bin/preprocess_change_order.py \
    --name cifs.tar.gz \
    --vocab_file voc.txt \
    --train_ratio 0.8 --val_ratio 0.1
```

### 5. Tokenize Data

```bash
python bin/tokenizer.py \
    --input_folder preprocessed_data/ \
    --vocab_file voc.txt
```

## Training

```bash
python bin/train.py \
    --data_dir tokenized_data/ \
    --out_dir out/model_name/ \
    --batch_size 32 \
    --max_iters 100000 \
    --learning_rate 3e-4 \
    --wandb_log true \
    --wandb_project crystalpropllm
```

### Key Training Parameters

- `batch_size`: Batch size (default: 32)
- `max_iters`: Maximum training iterations
- `learning_rate`: Initial learning rate (default: 3e-4)
- `n_layer`: Number of transformer layers (default: 6)
- `n_head`: Number of attention heads (default: 6)
- `n_embd`: Embedding dimension (default: 384)
- `compile`: Use PyTorch 2.0 compilation (default: true)

## Evaluation

### Standard Evaluation

```bash
python bin/evaluate.py \
    --input_folder preprocessed_data/ \
    --model out/model_name/ \
    --num_gens 3 \
    --temperature 0.8
```

### Changed Order Evaluation

```bash
python bin/evaluate_change_order.py \
    --input_folder preprocessed_data/ \
    --model out/model_name/ \
    --num_gens 3 \
    --temperature 0.8
```

### Evaluation Outputs

- Generated CIF structures
- Property predictions (CSV format)
- Performance metrics (R², MAE, RMSE)
- Scatter plots and residual plots
- Summary log file

## Prediction on New Data

```bash
# Basic prediction
python bin/predict.py \
    --input_cifs new_cifs.tar.gz \
    --model out/model_name/

# With features and operational conditions
python bin/predict.py \
    --input_cifs new_cifs.tar.gz \
    --features_json features.json \
    --properties_json properties.json \
    --model out/model_name/
```

## Benchmark Models

### Random Forest

```bash
cd benchmark/randomforest

# Train on electrical conductivity
python rf_sigma.py

# Train on Seebeck coefficient
python rf_seebeck.py

# Train on power factor
python rf_power_factor.py
```

### Multi-GNN

```bash
cd benchmark/multiGNN

# Train GNN model
python training.py

# Evaluate GNN model
python eval_model.py
```

## Data Format

### CIF Format

Standard CIF files with optional blocks:

- `FEATURE_`: Material features (composition, structure)
- `OPERATION_`: Operational conditions (temperature, pressure, carrier concentration)
- `TARGET_`: Target properties to predict

### Properties JSON Format

```json
[
  {
    "cif_id": "material_id",
    "condition": {
      "T": 300,
      "log_carrier_concentration": 20.0
    },
    "property": {
      "S_uV_per_K": 150.0,
      "log_sigma_over_tau_SI": 15.2,
      "log_PF_over_tau_SI": 12.5
    }
  }
]
```

### Features JSON Format

```json
{
  "material_id": {
    "band_gap": 1.2,
    "formation_energy_per_atom": -0.5,
    "AtomicWeight_mean": 50.0,
    "Electronegativity_mean": 2.0
  }
}
```

## Key Features

- **Flexible data ordering**: Support for different block orderings (structure-first or operation-first)
- **Feature extraction**: Magpie composition features and RDF structure features
- **Multi-property prediction**: Simultaneous prediction of multiple material properties
- **Comprehensive evaluation**: R², MAE, RMSE, correlation analysis
- **Benchmarking**: Comparison with Random Forest and GNN baselines
- **Structure validation**: Bond length and space group consistency checks

## Target Properties

Common thermoelectric properties:

- `S_uV_per_K`: Seebeck coefficient (μV/K)
- `log_sigma_over_tau_SI`: Electrical conductivity / relaxation time
- `log_PF_over_tau_SI`: Power factor / relaxation time
- `log_carrier_concentration`: Carrier concentration

## Example Workflow

### Complete Training Pipeline

1. **Prepare data**:

```bash
python bin/prepare_custom.py --input_dir data/cifs --output_tar_gz cif_mp.tar.gz
```

2. **Extract features** (optional):

```bash
python bin/extract_features.py --input cif_mp.tar.gz --feature_type magpie --output features.json
```

3. **Preprocess**:

```bash
python bin/preprocess.py --name cif_mp.tar.gz --vocab_file voc.txt --train_ratio 0.8 --val_ratio 0.1
```

4. **Tokenize**:

```bash
python bin/tokenizer.py --input_folder preprocessed_data/ --vocab_file voc.txt
```

5. **Train**:

```bash
python bin/train.py --data_dir tokenized_data/ --out_dir out/te_model/ --max_iters 100000
```

6. **Evaluate**:

```bash
python bin/evaluate.py --input_folder preprocessed_data/ --model out/te_model/
```

## Citation

If you use this code in your research, please cite the relevant paper.

## License

This project is licensed under the MIT License.

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.

## Acknowledgments

This project builds upon research in:

- Crystal structure prediction using transformers
- Thermoelectric materials discovery
- Multi-modal deep learning for materials science
