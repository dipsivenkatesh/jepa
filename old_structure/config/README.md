# JEPA Configuration Templates

This directory contains configuration templates for different use cases of JEPA models.

## Available Templates

### 1. `default_config.yaml`
General-purpose configuration suitable for most applications.
- **Encoder**: Transformer (512 dim)
- **Predictor**: MLP (1024 hidden, 512 output)
- **Use case**: General experimentation and research

### 2. `vision_config.yaml` 
Optimized for computer vision tasks.
- **Encoder**: CNN (256 dim)
- **Predictor**: MLP (512 hidden, 256 output)
- **Input**: RGB images (3 × 224 × 224)
- **Use case**: Image classification, object detection, visual representation learning

### 3. `nlp_config.yaml`
Designed for natural language processing tasks.
- **Encoder**: Transformer (768 dim)
- **Predictor**: Transformer (1024 hidden, 768 output)
- **Input**: Text sequences (vocab size 50k, seq length 512)
- **Use case**: Text modeling, language understanding, sequence prediction

### 4. `timeseries_config.yaml`
Tailored for time series analysis.
- **Encoder**: Transformer (128 dim)
- **Predictor**: MLP (256 hidden, 128 output)
- **Input**: Temporal sequences (32 features, 100 timesteps)
- **Use case**: Time series forecasting, sensor data analysis, temporal modeling

## Usage

### Using Pre-made Templates
```bash
# Copy and modify an existing template
cp config/vision_config.yaml my_vision_experiment.yaml
# Edit my_vision_experiment.yaml for your specific needs
```

### Generating New Templates
```bash
# Generate from CLI
python -m jepa.cli generate-config --output my_config.yaml --template vision
```

### Configuration Structure
All templates follow the same structure:
- `model`: Model architecture parameters
- `training`: Training hyperparameters
- `data`: Data loading and preprocessing settings
- General settings: device, seed, directories, etc.

## Customization

1. **Start with the closest template** to your use case
2. **Modify key parameters**:
   - `input_dim`: Match your data dimensions
   - `batch_size`: Adjust based on memory constraints
   - `learning_rate`: Tune for your specific task
   - `num_epochs`: Set based on your training budget
3. **Update paths**:
   - `train_data_path`, `val_data_path`, `test_data_path`
   - `output_dir`, `checkpoint_dir`
4. **Set experiment name** for organization

## Best Practices

- Keep original templates unchanged as references
- Use descriptive experiment names
- Version control your custom configurations
- Document any significant modifications
- Test with small datasets first
