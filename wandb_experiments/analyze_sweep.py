import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('sweep_results.csv')

print("Dataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nBasic statistics for avg_moves:")
print(df['avg_moves'].describe())

# Look at configurations with highest avg_moves
print("\nTop 20 configurations by avg_moves:")
top_configs = df.nlargest(20, 'avg_moves')[['n_blocks', 'n_hidden', 'n_channels', 'batch_size', 'optimizer', 'learning_rate_multiplier', 'avg_moves', 'final_test_loss', 'final_train_loss']]
print(top_configs)

# Analyze by architecture parameters
print("\n=== Analysis by Architecture Parameters ===")

print("\nAvg_moves by n_blocks:")
blocks_analysis = df.groupby('n_blocks')['avg_moves'].agg(['mean', 'std', 'count', 'max']).round(2)
print(blocks_analysis)

print("\nAvg_moves by n_hidden:")
hidden_analysis = df.groupby('n_hidden')['avg_moves'].agg(['mean', 'std', 'count', 'max']).round(2)
print(hidden_analysis)

print("\nAvg_moves by n_channels:")
channels_analysis = df.groupby('n_channels')['avg_moves'].agg(['mean', 'std', 'count', 'max']).round(2)
print(channels_analysis)

print("\nAvg_moves by optimizer:")
opt_analysis = df.groupby('optimizer')['avg_moves'].agg(['mean', 'std', 'count', 'max']).round(2)
print(opt_analysis)

print("\nAvg_moves by batch_size:")
batch_analysis = df.groupby('batch_size')['avg_moves'].agg(['mean', 'std', 'count', 'max']).round(2)
print(batch_analysis)

print("\nAvg_moves by learning_rate_multiplier:")
lr_analysis = df.groupby('learning_rate_multiplier')['avg_moves'].agg(['mean', 'std', 'count', 'max']).round(2)
print(lr_analysis)

# Look at correlation between avg_moves and loss
print("\n=== Correlation Analysis ===")
print(f"Correlation between avg_moves and final_test_loss: {df['avg_moves'].corr(df['final_test_loss']):.3f}")
print(f"Correlation between avg_moves and final_train_loss: {df['avg_moves'].corr(df['final_train_loss']):.3f}")

# Find configurations that perform significantly better than random
print("\n=== Configurations performing better than random ===")
# Assuming random would get ~20-30 moves on average
better_than_random = df[df['avg_moves'] > 30]
print(f"Configurations with avg_moves > 30: {len(better_than_random)}")

if len(better_than_random) > 0:
    print("\nBest performing configurations:")
    print(better_than_random.nlargest(10, 'avg_moves')[['n_blocks', 'n_hidden', 'n_channels', 'batch_size', 'optimizer', 'learning_rate_multiplier', 'avg_moves', 'final_test_loss']])

# Look at the move distribution patterns
print("\n=== Move Distribution Analysis ===")
move_cols = [col for col in df.columns if col.startswith('moves_hist_')]
print("Move histogram columns:", move_cols)

# Calculate average move distribution
avg_move_dist = df[move_cols].mean()
print("\nAverage move distribution across all experiments:")
for col in move_cols:
    print(f"{col}: {avg_move_dist[col]:.1f}")

# Find experiments with unusual move patterns
print("\nExperiments with games lasting 50+ moves:")
long_games = df[df['moves_hist_50-59'] + df['moves_hist_60-69'] + df['moves_hist_70-79'] + df['moves_hist_80-89'] + df['moves_hist_90-99'] > 10]
print(f"Count: {len(long_games)}")
if len(long_games) > 0:
    print(long_games[['n_blocks', 'n_hidden', 'n_channels', 'avg_moves', 'moves_hist_50-59', 'moves_hist_60-69', 'moves_hist_70-79']].head())