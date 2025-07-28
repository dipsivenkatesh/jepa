# JEPA CLI Documentation

The JEPA CLI provides a convenient command-line interface for training and evaluating JEPA (Joint Embedding Predictive Architecture) models.

## Installation

Make sure you have the JEPA package installed with all dependencies:

```bash
pip install -e .
```

## Quick Start

### 1. Generate a Configuration File

```bash
# Generate default configuration
python -m jepa.cli generate-config --output config.yaml

# Generate template for specific domains
python -m jepa.cli generate-config --output config/vision_config.yaml --template vision
python -m jepa.cli generate-config --output config/nlp_config.yaml --template nlp
python -m jepa.cli generate-config --output config/timeseries_config.yaml --template timeseries
```

### 2. Train a Model

```bash
python -m jepa.cli train \
    --config config.yaml \
    --train-data /path/to/train_data.npy \
    --val-data /path/to/val_data.npy \
    --experiment-name my_experiment \
    --num-epochs 100 \
    --wandb \
    --wandb-project my-jepa-project
```

### 3. Evaluate a Model

```bash
python -m jepa.cli evaluate \
    --model-path checkpoints/my_experiment/best_model.pt \
    --test-data /path/to/test_data.npy \
    --visualize \
    --compute-embeddings \
    --analyze-latent
```

## Commands

### `train`

Train a JEPA model with the specified configuration.

**Required Arguments:**
- Training data must be provided via config file or `--train-data`

**Optional Arguments:**
- `--config, -c`: Path to configuration YAML file
- `--train-data`: Path to training data
- `--val-data`: Path to validation data
- `--batch-size, -b`: Batch size for training
- `--learning-rate, -lr`: Learning rate
- `--num-epochs, -e`: Number of training epochs
- `--device`: Device to train on (auto/cuda/cpu)
- `--output-dir, -o`: Output directory for results
- `--checkpoint-dir`: Directory to save checkpoints
- `--experiment-name`: Name of the experiment
- `--resume`: Path to checkpoint to resume from
- `--generate-config`: Generate a default configuration file

**Wandb Arguments:**
- `--wandb`: Enable Weights & Biases logging
- `--wandb-project`: Wandb project name
- `--wandb-entity`: Wandb entity (username or team)
- `--wandb-name`: Wandb run name
- `--wandb-tags`: Wandb tags (space-separated)

**Example:**
```bash
python -m jepa.cli train \
    --config config/my_config.yaml \
    --train-data data/train.npy \
    --val-data data/val.npy \
    --experiment-name vision_experiment \
    --num-epochs 50 \
    --batch-size 64 \
    --device cuda \
    --wandb \
    --wandb-project jepa-experiments \
    --wandb-tags vision cnn
```

### `evaluate`

Evaluate a trained JEPA model.

**Required Arguments:**
- `--model-path, -m`: Path to trained model checkpoint

**Optional Arguments:**
- `--config, -c`: Path to configuration YAML file
- `--test-data`: Path to test data
- `--batch-size, -b`: Batch size for evaluation (default: 32)
- `--device`: Device to run evaluation on (default: auto)
- `--output-dir, -o`: Output directory for results (default: ./eval_results)
- `--save-predictions`: Save model predictions to file
- `--visualize`: Generate visualization plots
- `--compute-embeddings`: Compute and save embeddings
- `--analyze-latent`: Perform latent space analysis

**Example:**
```bash
python -m jepa.cli evaluate \
    --model-path checkpoints/vision_experiment/best_model.pt \
    --test-data data/test.npy \
    --output-dir eval_results/vision_experiment \
    --visualize \
    --compute-embeddings \
    --analyze-latent
```

### `generate-config`

Generate a configuration file template.

**Arguments:**
- `--output, -o`: Output path for configuration file (default: config.yaml)
- `--template`: Configuration template to use (default/vision/nlp/timeseries)

**Example:**
```bash
python -m jepa.cli generate-config \
    --output config/vision_config.yaml \
    --template vision
```

## Configuration File

The configuration file is a YAML file that specifies all the parameters for training and evaluation. Here's the structure:

```yaml
# Model configuration
model:
  encoder_type: "transformer"      # "transformer", "mlp", "cnn"
  encoder_dim: 512                 # Hidden dimension of encoder
  predictor_type: "mlp"            # "mlp", "transformer"
  predictor_hidden_dim: 1024       # Hidden dimension of predictor
  predictor_output_dim: 512        # Output dimension of predictor
  dropout: 0.1                     # Dropout rate

# Training configuration
training:
  batch_size: 32                   # Batch size
  learning_rate: 0.001             # Learning rate
  weight_decay: 0.0001             # Weight decay
  num_epochs: 100                  # Number of epochs
  warmup_epochs: 10                # Warmup epochs
  gradient_clip_norm: 1.0          # Gradient clipping
  save_every: 10                   # Save frequency
  early_stopping_patience: 20      # Early stopping
  log_interval: 100                # Logging frequency

# Data configuration
data:
  train_data_path: ""              # Training data path
  val_data_path: ""                # Validation data path
  test_data_path: ""               # Test data path
  num_workers: 4                   # Data loading workers
  pin_memory: true                 # Pin memory
  sequence_length: 10              # Sequence length
  input_dim: 784                   # Input dimension

# General configuration
device: "auto"                     # Device
seed: 42                          # Random seed
output_dir: "./outputs"           # Output directory
checkpoint_dir: "./checkpoints"   # Checkpoint directory
experiment_name: "jepa_experiment" # Experiment name

# Weights & Biases configuration
wandb:
  enabled: false                   # Enable wandb logging
  project: "jepa"                  # Wandb project name
  entity: null                     # Wandb entity (username/team)
  name: null                       # Run name (defaults to experiment_name)
  tags: null                       # List of tags
  notes: null                      # Run description
  log_model: true                  # Log model checkpoints as artifacts
  log_gradients: false             # Log gradients (expensive)
  log_freq: 100                    # Logging frequency
  watch_model: true                # Watch model architecture
```

## Data Format

The CLI expects data in NumPy array format (`.npy` files) with the following structure:

- **Training/Validation/Test Data**: Shape `(N, T, D)` where:
  - `N`: Number of samples
  - `T`: Sequence length (should match `data.sequence_length` in config)
  - `D`: Feature dimension (should match `data.input_dim` in config)

For JEPA training, the data should contain consecutive time steps where the model learns to predict `t+1` from `t`.

## Weights & Biases Integration

The JEPA CLI includes seamless integration with [Weights & Biases](https://wandb.ai) for experiment tracking, logging, and visualization.

### Setup

1. **Install wandb** (if not already installed):
```bash
pip install wandb
```

2. **Login to wandb**:
```bash
wandb login
```

### Usage

#### Via Configuration File
```yaml
wandb:
  enabled: true
  project: "my-jepa-project"
  entity: "my-team"
  tags: ["vision", "transformer"]
  notes: "Experimenting with JEPA on image data"
  log_model: true
  log_gradients: false
```

#### Via Command Line
```bash
python -m jepa.cli train \
    --config config.yaml \
    --wandb \
    --wandb-project my-jepa-project \
    --wandb-entity my-team \
    --wandb-tags vision transformer \
    --wandb-name experiment-1
```

### What Gets Logged

- **Training metrics**: Loss, learning rate, epoch progress
- **Validation metrics**: Validation loss, best loss tracking
- **System metrics**: GPU/CPU utilization, memory usage
- **Model architecture**: Network structure and parameters
- **Hyperparameters**: All configuration parameters
- **Model artifacts**: Best model checkpoints (optional)
- **Gradients**: Gradient histograms (optional)

### Wandb Features

- **Real-time monitoring**: Track training progress live
- **Hyperparameter sweeps**: Optimize hyperparameters automatically
- **Model comparison**: Compare multiple runs side-by-side
- **Collaboration**: Share experiments with team members
- **Reproducibility**: Full experiment tracking and code versioning

### Configuration Options

- `enabled`: Enable/disable wandb logging
- `project`: Project name for organizing experiments
- `entity`: Username or team name
- `name`: Custom run name (defaults to experiment_name)
- `tags`: List of tags for categorizing runs
- `notes`: Description of the experiment
- `log_model`: Save model checkpoints as wandb artifacts
- `log_gradients`: Log gradient histograms (can be memory intensive)
- `log_freq`: Frequency of detailed logging
- `watch_model`: Monitor model architecture and gradients

## Output Files

### Training Output

After training, you'll find the following files in the output directory:

```
outputs/my_experiment/
├── config.yaml              # Final configuration used
├── training_history.yaml    # Training metrics over time
└── logs/                    # Training logs

checkpoints/my_experiment/
├── best_model.pt            # Best model checkpoint
├── checkpoint_epoch_10.pt   # Periodic checkpoints
├── checkpoint_epoch_20.pt
└── ...
```

### Evaluation Output

After evaluation, you'll find:

```
eval_results/
├── eval_results.json        # Evaluation metrics
├── predictions.npy          # Model predictions (if --save-predictions)
├── targets.npy             # Ground truth targets (if --save-predictions)
├── embeddings.npy          # Learned embeddings (if --compute-embeddings)
├── evaluation_plots.png    # Visualization plots (if --visualize)
├── pca_analysis.png        # PCA analysis (if --analyze-latent)
├── latent_stats.json       # Latent space statistics (if --analyze-latent)
└── pca_stats.json          # PCA statistics (if --analyze-latent)
```

## Examples

See the `examples/cli_example.py` script for a complete example of using the CLI.

```bash
python examples/cli_example.py
```

## Tips

1. **Start with a template**: Use `generate-config` to create a template configuration for your domain
2. **Monitor training**: Check the training logs and use validation data to monitor progress
3. **Use GPU**: Set `device: cuda` in config or `--device cuda` for faster training
4. **Experiment names**: Use descriptive experiment names to organize your results
5. **Resume training**: Use `--resume` to continue training from a checkpoint
6. **Analyze results**: Use the evaluation options to understand your model's performance

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size in config
2. **Data not found**: Check file paths in config or CLI arguments
3. **Import errors**: Make sure the package is installed with `pip install -e .`
4. **Configuration errors**: Validate your YAML syntax and required fields

### Getting Help

```bash
# General help
python -m jepa.cli --help

# Command-specific help
python -m jepa.cli train --help
python -m jepa.cli evaluate --help
python -m jepa.cli generate-config --help
```
