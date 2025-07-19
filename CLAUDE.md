# Chess-SL: Supervised Learning Chess Engine

## Overview

Chess-SL is a supervised learning-based chess engine that uses artificial datasets to train a neural network to predict valid chess moves. The project implements a complete pipeline from data generation through model training to evaluation against random opponents.

## Project Architecture

### Core Components

#### 1. Data Representation (`chess_utils.py`)
- **Board Encoding**: Converts chess positions to 7-channel 8x8 tensors:
  - Channel 0: Turn indicator (+1 for white, -1 for black)
  - Channels 1-6: Piece positions (pawns, knights, bishops, rooks, queens, kings)
  - White pieces: +1, Black pieces: -1
- **Move Encoding**: Converts legal moves to 4096-dimensional binary vectors (64x64 action space)
- **Utility Functions**: Bidirectional conversion between chess representations and tensor formats

#### 2. Neural Network Model (`model.py`)
- **Architecture**: Convolutional neural network with configurable depth
- **Input**: 7x8x8 board representation tensor
- **Processing**: 
  - Configurable number of convolutional blocks (0-8 blocks)
  - Each block: Conv2d → BatchNorm2d → ReLU
  - Flatten → Fully connected hidden layer → Output layer
- **Output**: 4096-dimensional probability distribution over all possible moves
- **Activation**: Sigmoid output for move probability prediction

#### 3. Agent System (`agent.py`)
- **Base Agent**: Abstract interface for chess-playing agents
- **Random Agent**: Baseline that selects random legal moves
- **Model Agent**: Uses trained neural network for move selection via multinomial sampling
- **Move Validation**: Ensures selected moves are legal in current position

#### 4. Training Pipeline (`train.py`)
- **Data Loading**: Loads pre-generated position/move pairs from `games.pth`
- **Train/Test Split**: 80/20 split with configurable batch sizes
- **Loss Function**: Binary cross-entropy between predicted and actual legal moves
- **Optimization**: Support for SGD and Adam optimizers with configurable learning rates
- **Monitoring**: Real-time loss tracking and evaluation metrics

#### 5. Evaluation System (`evaluate.py`)
- **Model vs Random**: Evaluates trained model against random opponent
- **Model vs Model**: Head-to-head comparison between different models
- **Metrics**: Win rate, average game length, precision, recall
- **Game Simulation**: Complete game play with move validation and outcome tracking

#### 6. Data Generation (`gen_moves.py`)
- **Random Game Generation**: Creates training data through random game simulation
- **Position Extraction**: Captures board states and legal moves at each turn
- **Tensor Conversion**: Transforms game data into neural network input format
- **Data Persistence**: Saves processed datasets for training

### Configuration System (`config.py`)

Implements a flexible experiment configuration framework:

- **Hyperparameters**:
  - `N_HIDDEN`: Hidden layer size (1024-4096)
  - `N_BLOCKS`: Number of convolutional blocks (0-8)
  - `N_CHANNELS`: Convolutional filters (4-128)
  - `BATCH_SIZE`: Training batch size (64-256)
  - `NUM_EPOCHS`: Training epochs (20-50)
  - `LEARNING_RATE`: Optimizer learning rate multiplier

- **Training Parameters**:
  - `NUM_TRAINING_EXAMPLES`: Dataset size (10k-100k)
  - `OPTIMIZER`: SGD or Adam
  - `FILTER_SIZE`: Convolution kernel size (default: 3)

### Experiment Framework (`experiment.py`)

- **Grid Search**: Automated hyperparameter exploration
- **Run Management**: Timestamped experiment tracking
- **Results Storage**: CSV logging of all configurations and outcomes
- **Model Persistence**: Automatic saving of trained models
- **Development Mode**: Quick testing with reduced parameter sets

### Monitoring and Visualization

#### Observer System (`observer.py`)
- **Metric Tracking**: Real-time monitoring of training metrics
- **Data Export**: CSV and pickle file generation
- **Visualization**: Automatic plot generation for loss curves

#### Visualization Tools (`visualization.py`)
- **Board Visualization**: Multi-channel tensor display
- **Move Probability Maps**: Heatmaps of predicted move distributions
- **Interactive Chess Boards**: SVG rendering with move probability arrows

## Data Flow

1. **Data Generation** (`gen_moves.py`):
   - Generate random chess games
   - Extract position-move pairs
   - Convert to tensor format
   - Save as `games.pth`

2. **Training** (`train.py`):
   - Load position/move tensors
   - Split into train/test sets
   - Train neural network with binary cross-entropy loss
   - Monitor convergence metrics

3. **Evaluation** (`evaluate.py`):
   - Load trained model
   - Play games against random opponent
   - Calculate win rates, precision, recall
   - Generate performance statistics

4. **Experimentation** (`main.py`):
   - Coordinate full pipeline execution
   - Run hyperparameter experiments
   - Aggregate and store results

## Key Design Decisions

### Move Representation
- **64x64 Action Space**: Direct encoding of from-square to to-square moves
- **Binary Classification**: Each move treated as independent binary prediction
- **Promotion Handling**: Automatic queen promotion for pawn moves to back rank

### Network Architecture
- **Convolutional Processing**: Spatial feature extraction from board positions
- **Configurable Depth**: 0-8 conv blocks for architecture experimentation
- **Fully Connected Output**: Direct mapping to move probability space

### Training Strategy
- **Supervised Learning**: Train on human-interpretable legal move data
- **Binary Cross-Entropy**: Optimize probability matching for legal moves
- **Multi-Label Classification**: Predict all legal moves simultaneously

### Evaluation Methodology
- **Random Baseline**: Consistent comparison against random play
- **Game Completion**: Full games to terminal states
- **Statistical Metrics**: Precision/recall for move prediction accuracy

## Usage Patterns

### Training a Model
```bash
python main.py                    # Full hyperparameter grid search
python main.py --dev_mode         # Quick development testing
```

### Generating Training Data
```bash
python gen_moves.py               # Generate 1000 random games
```

### Direct Evaluation
```bash
python evaluate.py                # Evaluate specific model
```

## Project Structure
```
chess-sl/
├── main.py              # Main experiment coordinator
├── config.py            # Hyperparameter configurations
├── model.py             # Neural network architecture
├── chess_utils.py       # Chess-tensor conversion utilities
├── agent.py             # Chess playing agents
├── train.py             # Training pipeline
├── evaluate.py          # Model evaluation system
├── gen_moves.py         # Training data generation
├── experiment.py        # Experiment management framework
├── observer.py          # Metrics tracking system
├── visualization.py     # Visualization utilities
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## Technical Requirements
- **Python Libraries**: chess, torch, numpy, matplotlib
- **Hardware**: CUDA-compatible GPU recommended for training
- **Data**: Pre-generated `games.pth` file for training

## Recent Development
The project has been actively developed with recent commits focusing on:
- Addition of convolutional layers with configurable depth
- Experimentation with different numbers of blocks and filters
- Introduction of hidden layer size as a variable parameter
- Improved default configuration values
- Bug fixes for device placement issues
- Enhanced evaluation functions for comprehensive model assessment

This chess engine represents a supervised learning approach to chess AI, focusing on move prediction accuracy rather than game tree search, making it suitable for research into neural network architectures and training methodologies for chess position evaluation.