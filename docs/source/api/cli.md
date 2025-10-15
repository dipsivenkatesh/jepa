# CLI API

This section documents the command-line interface for JEPA.

## Main CLI Module

```{eval-rst}
.. automodule:: cli.__main__
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: cli.__main__.main
```

The main entry point for the CLI:

```bash
# Run CLI
python -m jepa.cli --help

# Or directly
python -m jepa.cli --help
```

## Training Commands

```{eval-rst}
.. automodule:: cli.train
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: cli.train.add_train_args

.. autofunction:: cli.train.train
```

### Basic Training

Train a JEPA model with default settings:

```bash
# Train with default configuration
python -m jepa.cli train

# Train with custom configuration
python -m jepa.cli train --config config/my_config.yaml

# Train with parameter overrides
python -m jepa.cli train \
  --config config/base.yaml \
  --epochs 100 \
  --batch-size 64 \
  --learning-rate 0.001
```

### Advanced Training Options

```bash
# Resume from checkpoint
python -m jepa.cli train \
  --config config/my_config.yaml \
  --resume checkpoints/latest.pth

# Distributed training
python -m jepa.cli train \
  --config config/large_config.yaml \
  --distributed \
  --world-size 4 \
  --rank 0

# Mixed precision training
python -m jepa.cli train \
  --config config/my_config.yaml \
  --mixed-precision \
  --gradient-accumulation 4
```

### Training with Different Backends

```bash
# Train with specific data format
python -m jepa.cli train \
  --data-type csv \
  --data-path data/train.csv \
  --data-columns "feat1,feat2,feat3"

# Train with JSON data
python -m jepa.cli train \
  --data-type json \
  --data-path data/sequences.json \
  --data-key "timeseries"

# Train with HuggingFace dataset
python -m jepa.cli train \
  --data-type huggingface \
  --dataset-name "imagenet-1k" \
  --dataset-split "train"
```

## Evaluation Commands

```{eval-rst}
.. automodule:: cli.evaluate
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: cli.evaluate.add_eval_args

.. autofunction:: cli.evaluate.evaluate
```

### Model Evaluation

Evaluate trained models:

```bash
# Basic evaluation
python -m jepa.cli evaluate \
  --model-path checkpoints/best_model.pth \
  --data-path data/test

# Evaluation with custom metrics
python -m jepa.cli evaluate \
  --model-path checkpoints/best_model.pth \
  --data-path data/test \
  --metrics "mse,cosine_similarity,accuracy"

# Generate predictions
python -m jepa.cli evaluate \
  --model-path checkpoints/best_model.pth \
  --data-path data/test \
  --output predictions.json \
  --save-embeddings
```

### Downstream Task Evaluation

```bash
# Linear probing
python -m jepa.cli evaluate \
  --model-path checkpoints/pretrained.pth \
  --task linear-probe \
  --labeled-data data/labeled_train.csv \
  --test-data data/labeled_test.csv

# Fine-tuning evaluation
python -m jepa.cli evaluate \
  --model-path checkpoints/pretrained.pth \
  --task fine-tune \
  --labeled-data data/labeled_train.csv \
  --epochs 20 \
  --learning-rate 1e-5
```

### Benchmark Evaluation

```bash
# Run standard benchmarks
python -m jepa.cli evaluate \
  --model-path checkpoints/model.pth \
  --benchmark vision-classification \
  --benchmark-datasets "cifar10,imagenet"

# Custom benchmark
python -m jepa.cli evaluate \
  --model-path checkpoints/model.pth \
  --benchmark custom \
  --benchmark-config benchmarks/my_benchmark.yaml
```

## Configuration Commands

### Configuration Management

```bash
# View default configuration
python -m jepa.cli config --show-default

# View specific configuration file
python -m jepa.cli config --show config/my_config.yaml

# Validate configuration
python -m jepa.cli config --validate config/my_config.yaml

# Create configuration template
python -m jepa.cli config --create-template vision my_vision_config.yaml
```

### Configuration Generation

```bash
# Generate configuration for different tasks
python -m jepa.cli config --create-template nlp nlp_config.yaml
python -m jepa.cli config --create-template timeseries ts_config.yaml
python -m jepa.cli config --create-template multimodal mm_config.yaml

# Generate configuration with specific parameters
python -m jepa.cli config --create-template vision \
  --model-size large \
  --training-type self-supervised \
  --output large_vision_config.yaml
```

## Data Commands

### Data Processing

```bash
# Validate data format
python -m jepa.cli data --validate \
  --data-type csv \
  --data-path data/train.csv

# Convert data formats
python -m jepa.cli data --convert \
  --input-type csv \
  --input-path data/train.csv \
  --output-type json \
  --output-path data/train.json

# Analyze data statistics
python -m jepa.cli data --analyze \
  --data-path data/train.csv \
  --output-report data_analysis.html
```

### Data Preprocessing

```bash
# Preprocess data
python -m jepa.cli data --preprocess \
  --data-path data/raw \
  --output-path data/processed \
  --transforms "normalize,augment,mask"

# Create data splits
python -m jepa.cli data --split \
  --data-path data/full_dataset.csv \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1
```

## Model Commands

### Model Management

```bash
# List available models
python -m jepa.cli model --list

# Show model information
python -m jepa.cli model --info checkpoints/model.pth

# Convert model format
python -m jepa.cli model --convert \
  --input checkpoints/model.pth \
  --output models/model.onnx \
  --format onnx

# Optimize model
python -m jepa.cli model --optimize \
  --input checkpoints/model.pth \
  --output checkpoints/optimized_model.pth \
  --optimization-level 2
```

### Model Analysis

```bash
# Analyze model architecture
python -m jepa.cli model --analyze checkpoints/model.pth

# Profile model performance
python -m jepa.cli model --profile \
  --model-path checkpoints/model.pth \
  --input-shape "3,224,224" \
  --batch-size 32

# Visualize model
python -m jepa.cli model --visualize \
  --model-path checkpoints/model.pth \
  --output model_architecture.pdf
```

## Experiment Commands

### Experiment Management

```bash
# List experiments
python -m jepa.cli experiment --list

# Show experiment details
python -m jepa.cli experiment --show experiment_001

# Compare experiments
python -m jepa.cli experiment --compare \
  --experiments "exp_001,exp_002,exp_003" \
  --metrics "loss,accuracy" \
  --output comparison.html

# Export experiment results
python -m jepa.cli experiment --export \
  --experiment exp_001 \
  --format csv \
  --output results.csv
```

### Hyperparameter Sweeps

```bash
# Run hyperparameter sweep
python -m jepa.cli sweep \
  --config config/sweep.yaml \
  --num-trials 100 \
  --optimize-metric val_loss

# Resume sweep
python -m jepa.cli sweep \
  --resume sweep_001 \
  --additional-trials 50

# Analyze sweep results
python -m jepa.cli sweep --analyze \
  --sweep-id sweep_001 \
  --output sweep_analysis.html
```

## CLI Utilities

```{eval-rst}
.. automodule:: cli.utils
   :members:
   :undoc-members:
   :show-inheritance:
```

### Setup Utilities

```{eval-rst}
.. autofunction:: cli.utils.setup_logging

.. autofunction:: cli.utils.validate_config

.. autofunction:: cli.utils.create_output_dir
```

Internal utilities for CLI operations:

```python
from jepa.cli.utils import setup_logging, validate_config

# Setup logging for CLI
logger = setup_logging(level="INFO", format="cli")

# Validate configuration before use
is_valid, errors = validate_config(config_path)
```

### Argument Parsing

```{eval-rst}
.. autofunction:: cli.utils.add_common_args

.. autofunction:: cli.utils.parse_overrides

.. autofunction:: cli.utils.merge_args_with_config
```

Utilities for handling command-line arguments:

```python
from jepa.cli.utils import parse_overrides

# Parse parameter overrides
overrides = parse_overrides([
    "--model.encoder_dim=1024",
    "--training.learning_rate=1e-4"
])
```

## Configuration Through CLI

### Parameter Overrides

Override any configuration parameter:

```bash
# Override top-level parameters
python -m jepa.cli train \
  --config config/base.yaml \
  --epochs 200 \
  --batch-size 128

# Override nested parameters
python -m jepa.cli train \
  --config config/base.yaml \
  --model.encoder_dim 1024 \
  --model.encoder_layers 24 \
  --training.optimizer.type adamw \
  --training.scheduler.type cosine
```

### Environment-Specific Configs

```bash
# Development mode
python -m jepa.cli train \
  --config config/base.yaml \
  --env dev \
  --debug

# Production mode
python -m jepa.cli train \
  --config config/base.yaml \
  --env prod \
  --distributed \
  --mixed-precision
```

## Advanced CLI Usage

### Batch Processing

```bash
# Process multiple datasets
for dataset in datasets/*.csv; do
  python -m jepa.cli train \
    --config config/base.yaml \
    --data-path "$dataset" \
    --output-dir "results/$(basename $dataset .csv)"
done

# Parallel training
python -m jepa.cli train --config config1.yaml &
python -m jepa.cli train --config config2.yaml &
python -m jepa.cli train --config config3.yaml &
wait
```

### Integration with Job Schedulers

```bash
# SLURM integration
sbatch --job-name=jepa_train \
       --output=logs/train_%j.out \
       --wrap="python -m jepa.cli train --config config/large.yaml"

# Kubernetes integration
kubectl run jepa-train \
  --image=jepa:latest \
  --restart=Never \
  --command -- python -m jepa.cli train --config /configs/k8s.yaml
```

### Monitoring and Logging

```bash
# Real-time monitoring
python -m jepa.cli train \
  --config config/base.yaml \
  --log-level DEBUG \
  --log-file logs/training.log \
  --progress-bar

# Remote logging
python -m jepa.cli train \
  --config config/base.yaml \
  --wandb-project my-project \
  --wandb-tags "experiment,baseline" \
  --tensorboard-dir logs/tb
```

## Error Handling and Debugging

### Common CLI Issues

```bash
# Check configuration syntax
python -m jepa.cli config --validate config/my_config.yaml

# Debug data loading
python -m jepa.cli data --validate --data-path data/train.csv --verbose

# Test model loading
python -m jepa.cli model --info checkpoints/model.pth --debug

# Dry run training
python -m jepa.cli train \
  --config config/base.yaml \
  --dry-run \
  --debug
```

### Verbose Output

```bash
# Enable verbose logging
python -m jepa.cli train \
  --config config/base.yaml \
  --verbose \
  --debug \
  --log-level DEBUG

# Profile execution
python -m jepa.cli train \
  --config config/base.yaml \
  --profile \
  --profile-output profile.html
```

## Examples

### Complete Training Pipeline

```bash
#!/bin/bash
# complete_pipeline.sh

# 1. Validate configuration
python -m jepa.cli config --validate config/experiment.yaml

# 2. Prepare data
python -m jepa.cli data --preprocess \
  --data-path data/raw \
  --output-path data/processed

# 3. Train model
python -m jepa.cli train \
  --config config/experiment.yaml \
  --data-path data/processed \
  --output-dir experiments/run_001

# 4. Evaluate model
python -m jepa.cli evaluate \
  --model-path experiments/run_001/best_model.pth \
  --data-path data/test \
  --output experiments/run_001/evaluation.json

# 5. Generate report
python -m jepa.cli experiment --export \
  --experiment run_001 \
  --format html \
  --output experiments/run_001/report.html
```

### Multi-Stage Training

```bash
# Stage 1: Self-supervised pretraining
python -m jepa.cli train \
  --config config/pretraining.yaml \
  --data-path data/unlabeled \
  --output-dir pretrain/

# Stage 2: Fine-tuning
python -m jepa.cli train \
  --config config/finetuning.yaml \
  --pretrained-model pretrain/best_model.pth \
  --data-path data/labeled \
  --output-dir finetune/

# Stage 3: Evaluation
python -m jepa.cli evaluate \
  --model-path finetune/best_model.pth \
  --task downstream \
  --data-path data/test
```

For more CLI examples and detailed usage, see the [CLI Examples](../examples/cli.md) and [Quick Start Guide](../guides/quickstart.md).
