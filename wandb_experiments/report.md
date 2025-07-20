# Chess-SL Hyperparameter Sweep Analysis Report

## Raw Data Analysis

### Architecture Impact by n_blocks
         avg_moves              final_test_loss
              mean    max count            mean
n_blocks                                       
0           13.804  21.99    34           0.032
1           22.808  36.31    30           0.122
4           35.484  50.68    48           0.044
8           38.540  63.67   148           0.034

### Architecture Impact by n_channels
           avg_moves              final_test_loss
                mean    max count            mean
n_channels                                       
4             12.705  27.98    24           0.116
16            11.191  32.92    20           0.253
64            30.071  50.68    32           0.052
128           38.423  63.67   184           0.013

### Training Configuration Impact by Optimizer
          avg_moves                final_test_loss
               mean    max     std            mean
optimizer                                         
ADAM         36.646  63.67  11.158           0.003
SGD           4.409  19.17   6.438           0.373

### Learning Rate Multiplier Impact
                         avg_moves                final_test_loss
                              mean    max     std            mean
learning_rate_multiplier                                         
1                           31.264  63.67  16.414           0.060
3                           35.193  54.07  12.179           0.026

### Top 10 Performing Models
     n_blocks  n_hidden  n_channels optimizer  learning_rate_multiplier  batch_size  avg_moves  final_test_loss
155         8      1024         128      ADAM                         1          64      63.67         0.001457
60          8      1024         128      ADAM                         1          64      59.48         0.001218
65          8      1024         128      ADAM                         1          64      58.59         0.001255
134         8      1024         128      ADAM                         1          64      58.48         0.001262
168         8      1024         128      ADAM                         1          64      58.03         0.001154
119         8      1024         128      ADAM                         1         128      54.86         0.001193
153         8      1024         128      ADAM                         3         256      54.07         0.001387
78          8      1024         128      ADAM                         1          64      53.38         0.001355
101         8      1024         128      ADAM                         1          64      51.46         0.001462
64          8      1024         128      ADAM                         1          64      50.82         0.001502

### Performance Thresholds
Models with avg_moves > 45: 47
Models with avg_moves < 20: 49

### Good Models (>45 avg_moves) Characteristics
   n_blocks  n_channels optimizer  count
2         8         128      ADAM     40
1         4         128      ADAM      6
0         4          64      ADAM      1

### Poor Models (<20 avg_moves) Characteristics
    n_blocks  n_channels optimizer  count
1          0           4       SGD      5
3          0          16       SGD      4
6          0         128       SGD      4
19         8          16       SGD      4
2          0          16      ADAM      3
4          0          64      ADAM      3
5          0         128      ADAM      3
0          0           4      ADAM      2
20         8         128      ADAM      2
17         8           4      ADAM      2
16         4          64       SGD      2
11         1         128      ADAM      2
9          1          16       SGD      2
8          1           4       SGD      2
21         8         128       SGD      2
12         1         128       SGD      1
13         4           4      ADAM      1
14         4           4       SGD      1
15         4          16       SGD      1
10         1          64       SGD      1
18         8           4       SGD      1
7          1           4      ADAM      1

### Loss vs Performance Categories
                mean  count
loss_category              
Very Low       35.33    242
Low             0.55     13
Medium          0.59      3
High            0.59      1
Very High       0.43      1

### Top Configuration Combinations (with at least 5 runs)
    n_blocks  n_channels optimizer       mean  count       std
25         8         128      ADAM  42.044444    117  8.222186
19         4         128      ADAM  40.526286     35  5.391876
17         4          64      ADAM  37.313333      6  9.782005
24         8          64      ADAM  36.010556     18  5.886058
11         1         128      ADAM  30.437778     18  5.972772
7          1           4      ADAM  22.213333      6  2.961842
5          0         128      ADAM  17.925714      7  6.621696
4          0          64      ADAM  17.704000      5  3.507710
0          0           4      ADAM  15.288000      5  7.746852
1          0           4       SGD   7.272000      5  6.553512

### Parameter Efficiency (avg_moves per million parameters)
                   mean   max
n_blocks n_hidden            
8        1024      3.27  5.26
4        1024      3.07  4.84
0        1024      2.92  4.70
1        1024      2.28  4.92
8        2048      1.57  2.61
0        2048      1.53  2.28
4        2048      1.53  1.66
1        2048      0.87  2.39
         4096      0.80  1.57
4        4096      0.79  1.57
8        4096      0.79  1.19
0        4096      0.74  1.18

## Key Findings

The data reveals clear patterns in what makes a chess model perform well. Out of 260 experiments, the performance ranges from 0.41 to 63.67 average moves before predicting an illegal move.

### Architecture Matters Most

The number of convolutional blocks is critical. Models with 8 blocks average 38.5 moves compared to just 13.8 moves for models with no convolution layers. This progression is nearly monotonic - more depth consistently helps.

Channel count is equally important. Models with 128 channels average 38.4 moves while those with only 4 channels manage just 12.7 moves. The jump from 64 to 128 channels provides a significant boost.

### Optimizer Choice is Decisive

ADAM completely dominates SGD. ADAM models average 36.6 moves while SGD models only manage 4.4 moves. This 8x difference makes optimizer choice the most important single decision.

### The Winning Formula

Looking at the top 10 models, they all share the same pattern:
- 8 convolutional blocks
- 128 channels 
- ADAM optimizer
- 1024 hidden units
- Mostly batch size 64

The best single model achieves 63.67 average moves with this exact configuration plus a 1x learning rate multiplier.

### Performance Distribution

47 models (18%) achieve >45 average moves and are considered high performers. Of these, 40 use the winning formula of 8 blocks + 128 channels + ADAM. Only 6 high performers use different architectures (4 blocks + 128 channels).

49 models (19%) perform poorly with <20 average moves. These failures come from using SGD, having too few channels, or lacking convolutional layers entirely.

### Learning Rate Effects

Higher learning rates (3x multiplier) improve average performance across all models, but the absolute best model uses a 1x multiplier. This suggests the optimal learning rate depends on other hyperparameters.

### Efficiency Analysis

The 8 blocks + 1024 hidden configuration provides the best efficiency at 3.27 moves per million parameters. Larger models with 4096 hidden units are much less efficient at only 0.79 moves per million parameters.

### Failure Modes

Even optimal architectures can fail. Two models with the winning 8 blocks + 128 channels + ADAM configuration still performed poorly, suggesting training instability or random seed effects matter.

SGD fails consistently regardless of architecture. Even with 8 blocks and 128 channels, SGD models can't exceed 19 moves on average.

## Recommendations

For maximum performance: Use 8 convolutional blocks, 128 channels, ADAM optimizer, 1024 hidden units, batch size 64, and start with 1x learning rate multiplier.

For resource constraints: The minimum viable configuration is 4 blocks, 64 channels, ADAM optimizer. This should achieve around 37 average moves.

Avoid: SGD optimizer, fewer than 64 channels, or fewer than 4 convolutional blocks.

The data shows that chess move prediction requires sufficient model capacity (channels and depth) combined with proper optimization (ADAM). The 85% concentration of top performers in one architectural configuration provides strong evidence that there's an optimal design for this task.