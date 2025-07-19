# Chess-SL with Weights & Biases

This directory contains a WandB-enabled version of the Chess-SL experiment framework, replacing the custom experiment system with Weights & Biases for better experiment tracking, visualization, and hyperparameter optimization.

## Setup Instructions

### 1. Install Dependencies

```bash
cd wandb_experiments
pip install -r requirements.txt
```

### 2. Set up Weights & Biases

First, create a free account at [wandb.ai](https://wandb.ai) if you don't have one.

Then login from the command line:
```bash
wandb login
```

This will open your browser to get an API key. Copy and paste it when prompted.

### 3. Generate Training Data

The experiments need the `games.pth` training data file. You can either:

**Option A: Copy from parent directory (if it exists):**
```bash
cp ../games.pth .
```

**Option B: Generate new training data:**
```bash
python gen_moves.py
```

This will generate 1000 random chess games and save them as `games.pth`.

## Running Experiments

### Full Hyperparameter Sweep

Run the complete grid search over all hyperparameters (this will take a long time):

```bash
python main.py --project chess-sl-full
```

### Development Mode (Recommended for Testing)

Run a smaller parameter sweep for quick testing:

```bash
python main.py --dev_mode --project chess-sl-dev
```

### Single Run

Run just one experiment with default parameters:

```bash
python main.py --single_run --project chess-sl-test
```

### Custom Project/Team

If you want to organize experiments under a specific project or team:

```bash
python main.py --dev_mode --project my-chess-experiments --entity my-team
```

## Parameter Configurations

### Full Sweep Parameters
- **num_epochs**: [20, 50]
- **n_blocks**: [0, 1, 4, 8] (number of conv blocks)
- **n_channels**: [4, 16, 64, 128] (conv filters)
- **batch_size**: [64, 128, 256]
- **learning_rate_multiplier**: [1, 3]
- **optimizer**: ['SGD', 'ADAM']
- **n_hidden**: [1024, 4096]
- **num_training_examples**: [10000, 30000, 100000]

### Development Mode Parameters (Reduced)
- **num_epochs**: [10]
- **n_blocks**: [0, 1]
- **n_channels**: [8, 16]
- **batch_size**: [128]
- **learning_rate_multiplier**: [1]
- **optimizer**: ['ADAM']
- **n_hidden**: [1024]
- **num_training_examples**: [10000]

## Monitoring Results

### WandB Dashboard

Once you start an experiment, you can monitor progress at:
```
https://wandb.ai/your-username/your-project-name
```

The dashboard will show:
- Real-time training/test loss curves
- Hyperparameter importance analysis
- Model performance metrics (win rate, precision, recall)
- System metrics (GPU usage, etc.)

### Key Metrics Tracked

**Training Metrics:**
- `train_loss`: Training loss per epoch
- `test_loss`: Validation loss per epoch
- `batch_loss`: Loss per training batch
- `best_test_loss`: Best validation loss achieved

**Model Performance:**
- `score`: Total wins against random opponent (out of 100 games)
- `win_rate`: Win percentage against random opponent
- `avg_moves`: Average game length
- `precision`: Move prediction precision
- `recall`: Move prediction recall

**Model Info:**
- `model_parameters`: Total number of parameters
- `actual_learning_rate`: Computed learning rate used

## Differences from Original

### Advantages of WandB Version:
1. **Better Visualization**: Interactive plots and dashboards
2. **Experiment Comparison**: Easy side-by-side comparison of runs
3. **Hyperparameter Analysis**: Automatic importance and correlation analysis
4. **Model Artifacts**: Automatic model versioning and storage
5. **Collaboration**: Easy sharing and team access
6. **Mobile Access**: Monitor experiments from your phone

### Code Changes:
- Replaced custom `Config` and `Experiment` classes with WandB sweep configuration
- Added real-time logging throughout training
- Automatic model artifact saving
- Simplified configuration management
- Better error handling and progress reporting

## File Structure

```
wandb_experiments/
├── README.md           # This file
├── requirements.txt    # Python dependencies
├── config.py          # WandB sweep configurations
├── main.py            # Main experiment runner
├── train.py           # Training with WandB logging
├── model.py           # Neural network (parameter injection)
├── gen_moves.py       # Training data generation
├── chess_utils.py     # Chess utilities (copied from parent)
├── agent.py           # Chess agents (copied from parent)
├── evaluate.py        # Model evaluation (copied from parent)
└── games.pth          # Training data (generated/copied)
```

## Tips

1. **Start with development mode** to test everything works before running full sweeps
2. **Use descriptive project names** to organize different experiment types
3. **Monitor the dashboard** to stop underperforming sweeps early
4. **Check your WandB usage** - free accounts have limits on parallel runs and storage