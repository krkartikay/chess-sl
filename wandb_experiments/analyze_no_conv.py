import pandas as pd
import numpy as np

# Set pandas options to display full dataframes
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

df = pd.read_csv('sweep_results.csv')

print("=== ANALYSIS OF MODELS WITH NO CONVOLUTIONAL LAYERS (n_blocks = 0) ===")

# Filter for models with no conv layers
no_conv = df[df['n_blocks'] == 0].copy()

print(f"\nTotal models with 0 convolutional blocks: {len(no_conv)}")
print(f"Performance range: {no_conv['avg_moves'].min():.2f} to {no_conv['avg_moves'].max():.2f} avg_moves")
print(f"Mean performance: {no_conv['avg_moves'].mean():.2f} ± {no_conv['avg_moves'].std():.2f}")

print("\n=== DETAILED BREAKDOWN ===")

print("\nPerformance by optimizer:")
opt_analysis = no_conv.groupby('optimizer')['avg_moves'].agg(['count', 'mean', 'std', 'min', 'max']).round(2)
print(opt_analysis)

print("\nPerformance by hidden layer size:")
hidden_analysis = no_conv.groupby('n_hidden')['avg_moves'].agg(['count', 'mean', 'std', 'min', 'max']).round(2)
print(hidden_analysis)

print("\nPerformance by n_channels (which becomes irrelevant but still set):")
channels_analysis = no_conv.groupby('n_channels')['avg_moves'].agg(['count', 'mean', 'std', 'min', 'max']).round(2)
print(channels_analysis)

print("\nPerformance by batch size:")
batch_analysis = no_conv.groupby('batch_size')['avg_moves'].agg(['count', 'mean', 'std', 'min', 'max']).round(2)
print(batch_analysis)

print("\nPerformance by learning rate multiplier:")
lr_analysis = no_conv.groupby('learning_rate_multiplier')['avg_moves'].agg(['count', 'mean', 'std', 'min', 'max']).round(2)
print(lr_analysis)

print("\n=== TOP PERFORMING NO-CONV MODELS ===")
top_no_conv = no_conv.nlargest(10, 'avg_moves')[['n_hidden', 'optimizer', 'batch_size', 'learning_rate_multiplier', 'avg_moves', 'final_test_loss', 'final_train_loss']]
print(top_no_conv)

print("\n=== BOTTOM PERFORMING NO-CONV MODELS ===")
bottom_no_conv = no_conv.nsmallest(10, 'avg_moves')[['n_hidden', 'optimizer', 'batch_size', 'learning_rate_multiplier', 'avg_moves', 'final_test_loss', 'final_train_loss']]
print(bottom_no_conv)

print("\n=== CONFIGURATION COMBINATIONS ===")
config_combos = no_conv.groupby(['n_hidden', 'optimizer', 'learning_rate_multiplier'])['avg_moves'].agg(['count', 'mean', 'std']).round(2)
print(config_combos)

print("\n=== LOSS ANALYSIS ===")
print(f"Correlation between avg_moves and final_test_loss: {no_conv['avg_moves'].corr(no_conv['final_test_loss']):.3f}")
print(f"Correlation between avg_moves and final_train_loss: {no_conv['avg_moves'].corr(no_conv['final_train_loss']):.3f}")

print("\nTest loss distribution:")
loss_bins = pd.cut(no_conv['final_test_loss'], bins=5)
loss_dist = no_conv.groupby(loss_bins)['avg_moves'].agg(['count', 'mean']).round(2)
print(loss_dist)

print("\n=== PARAMETER EFFICIENCY (for no-conv models) ===")
no_conv['moves_per_million_params'] = no_conv['avg_moves'] / (no_conv['model_parameters'] / 1000000)
efficiency = no_conv.groupby('n_hidden')['moves_per_million_params'].agg(['mean', 'max', 'count']).round(2)
print("Efficiency by hidden layer size:")
print(efficiency)

print("\n=== MODEL ARCHITECTURE ANALYSIS ===")
print("Since there are no conv layers, the model is essentially:")
print("Input (7x8x8=448) -> Flatten -> FC(n_hidden) -> FC(4096)")

print(f"\nParameter counts by hidden size:")
for hidden in no_conv['n_hidden'].unique():
    subset = no_conv[no_conv['n_hidden'] == hidden]
    if len(subset) > 0:
        params = subset['model_parameters'].iloc[0]
        print(f"n_hidden={hidden}: {params:,} parameters")

print(f"\n=== WHAT MATTERS FOR NO-CONV MODELS ===")
print("Best configuration for no-conv models:")
best_no_conv = no_conv.loc[no_conv['avg_moves'].idxmax()]
print(f"n_hidden: {best_no_conv['n_hidden']}")
print(f"optimizer: {best_no_conv['optimizer']}")
print(f"batch_size: {best_no_conv['batch_size']}")
print(f"learning_rate_multiplier: {best_no_conv['learning_rate_multiplier']}")
print(f"avg_moves: {best_no_conv['avg_moves']}")
print(f"final_test_loss: {best_no_conv['final_test_loss']}")

print(f"\nWorst configuration for no-conv models:")
worst_no_conv = no_conv.loc[no_conv['avg_moves'].idxmin()]
print(f"n_hidden: {worst_no_conv['n_hidden']}")
print(f"optimizer: {worst_no_conv['optimizer']}")
print(f"batch_size: {worst_no_conv['batch_size']}")
print(f"learning_rate_multiplier: {worst_no_conv['learning_rate_multiplier']}")
print(f"avg_moves: {worst_no_conv['avg_moves']}")
print(f"final_test_loss: {worst_no_conv['final_test_loss']}")