# Interactive Chess Model Visualization

This directory contains comprehensive web-based tools for exploring and visualizing your trained Chess-SL models interactively in the browser.

## Features

### 🏆 Chess Model Explorer (`chess_model_viz.py`)
- **Interactive Chess Board**: Visual board with move probability arrows
- **Real-time Analysis**: Load any position (FEN, random, starting)
- **Multiple Visualizations**:
  - Move probability heatmaps (64x64 from/to squares)
  - Legal moves analysis with probability rankings
  - Top-K move predictions with filtering
- **Model Comparison**: Side-by-side architecture analysis
- **Detailed Statistics**: Precision metrics and probability distributions

### 🎮 Game Explorer (`game_explorer.py`)
- **PGN Game Analysis**: Upload and analyze complete chess games
- **Move-by-Move Evaluation**: Track model accuracy throughout games
- **Famous Games Database**: Pre-loaded classical games
- **Performance Metrics**:
  - Top-1, Top-3 accuracy tracking
  - Move ranking analysis
  - Probability trends over time
- **Position Explorer**: Deep-dive into specific game positions

## Setup Instructions

### 1. Install Dependencies
```bash
cd interactive_chess_viz
pip install -r requirements.txt
```

### 2. Ensure Model Availability
The tools automatically scan for `.pt` model files in the parent directories. Make sure you have trained models available from either:
- `../wandb_experiments/` (WandB experiments)
- `../results/` (Original experiment results)
- Any `.pt` files containing model weights

### 3. Launch Applications

**Model Explorer:**
```bash
streamlit run chess_model_viz.py
```

**Game Explorer:**
```bash
streamlit run game_explorer.py
```

Both will open in your browser at `http://localhost:8501` and `http://localhost:8502`.

## Usage Guide

### Chess Model Explorer

1. **Select Model**: Choose from available `.pt` files
2. **Configure Architecture**: Set model parameters (blocks, channels, hidden size)
3. **Input Position**: 
   - Paste FEN strings
   - Use starting position
   - Generate random positions
4. **Analyze**: View move probabilities, heatmaps, and legal move rankings
5. **Explore**: Adjust Top-K moves, enable/disable visualizations

### Game Explorer

1. **Load Model**: Same model selection as above
2. **Input Game**:
   - Paste PGN text directly
   - Choose from famous games (Morphy, Kasparov, etc.)
   - Upload `.pgn` files
3. **Analyze Game**: Model evaluates every move in the game
4. **Review Results**:
   - Overall accuracy statistics
   - Move probability trends
   - Detailed move-by-move analysis
   - Position-specific exploration

## Key Visualizations

### Move Probability Arrows
- **Green arrows** on chess board show predicted moves
- **Arrow thickness/opacity** indicates probability strength
- **Interactive filtering** by Top-K most likely moves

### Heatmap Analysis
- **64x64 matrix** showing all possible move probabilities
- **Hover tooltips** with square names and exact probabilities
- **Color intensity** represents prediction confidence

### Performance Charts
- **Accuracy over time**: How well model predicts actual moves
- **Move ranking trends**: Track model consistency
- **Legal vs illegal** probability distributions

### Statistical Dashboards
- **Real-time metrics**: Precision, recall, probability sums
- **Comparative analysis**: Multiple models side-by-side
- **Interactive filtering**: Focus on interesting positions/moves

## Libraries Used

### Core Framework
- **Streamlit**: Web app framework for rapid ML interfaces
- **python-chess**: Chess logic, board representation, PGN parsing
- **PyTorch**: Model loading and inference

### Visualization
- **Plotly**: Interactive charts and heatmaps
- **chess.svg**: SVG chess board rendering with arrows
- **Matplotlib**: Additional plotting capabilities

### Data Processing
- **NumPy**: Numerical operations and tensor manipulation
- **Pandas**: Data analysis for move statistics

## Technical Architecture

### Model Loading
- **Cached models**: `@st.cache_resource` for efficient reloading
- **GPU support**: Automatic CUDA detection and model placement
- **Configuration injection**: Dynamic architecture parameters

### Real-time Inference
- **Batch processing**: Efficient tensor operations
- **Move validation**: Integration with chess rules engine
- **Probability mapping**: Action space to chess move conversion

### Interactive Features
- **State management**: Streamlit session state for position history
- **Real-time updates**: Dynamic recomputation on parameter changes
- **Multi-page navigation**: Separate apps for different analysis types

## Extending the Tools

### Adding New Visualizations
1. Create new plotting functions in the respective files
2. Add UI controls for new features in the sidebar
3. Integrate with existing model inference pipeline

### Supporting New Model Architectures
1. Update `model.py` with new architecture classes
2. Modify model loading functions to handle new parameters
3. Add UI controls for new hyperparameters

### Adding Analysis Features
1. Implement new analysis functions (e.g., opening analysis)
2. Create corresponding visualization components
3. Add to the main application workflow

## Performance Tips

1. **Model caching**: Models are cached automatically - restart app to reload
2. **Position caching**: Complex positions are memoized for faster recomputation
3. **GPU usage**: Enable CUDA for faster inference on large models
4. **Batch processing**: Analyze multiple positions simultaneously when possible

## Troubleshooting

### Common Issues

**"No model files found"**
- Ensure you have trained models with `.pt` extension
- Check that files are accessible from the visualization directory

**"Error loading model"**
- Verify model architecture matches the configuration
- Check that model was saved with compatible PyTorch version

**"Invalid FEN"**
- Ensure FEN string follows standard format
- Use chess position editors to generate valid FENs

**Slow performance**
- Use smaller models for interactive exploration
- Enable GPU acceleration if available
- Reduce Top-K parameter for faster visualization

## Future Enhancements

- **Multi-model comparison**: Side-by-side model analysis
- **Opening book integration**: Opening move analysis
- **Engine comparison**: Compare with Stockfish evaluations
- **Training visualization**: Real-time training progress
- **Tournament analysis**: Batch game processing
- **Export capabilities**: Save analysis results and visualizations