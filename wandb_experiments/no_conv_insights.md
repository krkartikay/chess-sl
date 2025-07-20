# Analysis of Models with No Convolutional Layers

## Key Findings for No-Conv Models (n_blocks = 0)

From 34 models with no convolutional layers, we can see what happens when the chess model is purely a feedforward network without spatial feature extraction.

### Performance Range
- Best: 21.99 avg_moves  
- Worst: 0.43 avg_moves
- Mean: 13.80 ± 7.51 avg_moves
- This is dramatically lower than models with convolution (max 63.67)

### Architecture Details
Without conv layers, the model becomes:
Input (7×8×8=448 features) → Flatten → FC(n_hidden) → FC(4096)

Parameter counts:
- n_hidden=1024: 4.66M parameters
- n_hidden=2048: 9.31M parameters  
- n_hidden=4096: 18.62M parameters

### What Still Matters Without Convolution

**Optimizer choice remains critical:**
- ADAM: 16.51 ± 6.48 avg_moves (21 models)
- SGD: 9.44 ± 7.19 avg_moves (13 models)

Even without spatial processing, ADAM is 75% better than SGD.

**Hidden layer size has minimal impact:**
- 1024: 13.62 ± 8.39 avg_moves
- 2048: 14.25 ± 6.68 avg_moves
- 4096: 13.80 ± 7.19 avg_moves

Surprisingly, more parameters don't help much in the no-conv setting.

**Batch size preferences:**
- 64: 15.22 ± 7.47 avg_moves (best)
- 128: 11.58 ± 8.44 avg_moves
- 256: 11.88 ± 7.23 avg_moves

Smaller batches work better even without convolution.

**Learning rate effects:**
- 1x: 14.23 ± 7.35 avg_moves (28 models)
- 3x: 11.84 ± 8.67 avg_moves (6 models)

Unlike conv models, higher learning rates hurt performance here.

### Best vs Worst No-Conv Models

**Best performing (21.99 avg_moves):**
- n_hidden: 4096
- optimizer: ADAM
- batch_size: 64
- learning_rate_multiplier: 1
- final_test_loss: 0.003

**Worst performing (0.43 avg_moves):**
- n_hidden: 1024
- optimizer: SGD  
- batch_size: 64
- learning_rate_multiplier: 3
- final_test_loss: 0.575

### Efficiency Analysis
Moves per million parameters:
- 1024 hidden: 2.92 (most efficient)
- 2048 hidden: 1.53
- 4096 hidden: 0.74 (least efficient)

The 1024 hidden model is 4x more efficient than 4096.

### Key Insights

1. **Spatial reasoning matters enormously**: The best no-conv model (21.99) can't compete with even mediocre conv models (30+ avg_moves).

2. **Optimizer choice transcends architecture**: ADAM vs SGD remains the most important decision even in simple feedforward networks.

3. **Parameter scaling doesn't help without structure**: Going from 4.7M to 18.6M parameters (4x increase) provides no meaningful improvement.

4. **Learning rate sensitivity differs**: No-conv models prefer conservative learning rates (1x) while conv models benefit from higher rates (3x).

5. **Performance ceiling**: Even optimal configurations can't exceed ~22 avg_moves without spatial processing.

### What This Tells Us About Chess

Chess position evaluation fundamentally requires spatial pattern recognition. A feedforward network can learn some basic rules (piece values, simple tactics) but cannot develop the spatial understanding needed for strong play.

The fact that even a 18.6M parameter feedforward network tops out at 22 moves while an 13.6M parameter conv network reaches 63+ moves demonstrates that architecture matters more than raw parameter count for this domain.

This validates the original design choice to use convolutional layers for chess - the spatial structure of the board is not just helpful but essential for meaningful performance.