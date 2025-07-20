import pandas as pd
import numpy as np

# Set pandas options to display full dataframes
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

df = pd.read_csv('sweep_results.csv')

print("=== KEY INSIGHTS FOR REPORT ===")

# 1. Architecture insights
print("\n1. ARCHITECTURE IMPACT:")
print("Best n_blocks configurations:")
blocks_perf = df.groupby('n_blocks').agg({
    'avg_moves': ['mean', 'max', 'count'],
    'final_test_loss': 'mean'
}).round(3)
print(blocks_perf)

print("\nBest n_channels configurations:")
channels_perf = df.groupby('n_channels').agg({
    'avg_moves': ['mean', 'max', 'count'], 
    'final_test_loss': 'mean'
}).round(3)
print(channels_perf)

# 2. Training configuration insights
print("\n2. TRAINING CONFIGURATION IMPACT:")
print("Optimizer comparison:")
opt_perf = df.groupby('optimizer').agg({
    'avg_moves': ['mean', 'max', 'std'],
    'final_test_loss': 'mean'
}).round(3)
print(opt_perf)

print("\nLearning rate multiplier impact:")
lr_perf = df.groupby('learning_rate_multiplier').agg({
    'avg_moves': ['mean', 'max', 'std'],
    'final_test_loss': 'mean'
}).round(3)
print(lr_perf)

# 3. Best performing combination analysis
print("\n3. BEST PERFORMING COMBINATIONS:")
top_10 = df.nlargest(10, 'avg_moves')
best_configs = top_10[['n_blocks', 'n_hidden', 'n_channels', 'optimizer', 'learning_rate_multiplier', 'batch_size', 'avg_moves', 'final_test_loss']]
print(best_configs)

# 4. What separates good from bad models?
print("\n4. PERFORMANCE THRESHOLDS:")
good_models = df[df['avg_moves'] > 45]  # Top quartile
poor_models = df[df['avg_moves'] < 20]  # Bottom quartile

print(f"Models with avg_moves > 45: {len(good_models)}")
print(f"Models with avg_moves < 20: {len(poor_models)}")

print("\nGood models (>45 avg_moves) characteristics:")
good_summary = good_models.groupby(['n_blocks', 'n_channels', 'optimizer']).size().reset_index(name='count')
print(good_summary.sort_values('count', ascending=False))

print("\nPoor models (<20 avg_moves) characteristics:")
poor_summary = poor_models.groupby(['n_blocks', 'n_channels', 'optimizer']).size().reset_index(name='count')
print(poor_summary.sort_values('count', ascending=False))

# 5. Loss vs performance relationship
print("\n5. LOSS VS PERFORMANCE:")
df['loss_category'] = pd.cut(df['final_test_loss'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
loss_perf = df.groupby('loss_category')['avg_moves'].agg(['mean', 'count']).round(2)
print(loss_perf)

# 6. Identify clear winners
print("\n6. CLEAR WINNING CONFIGURATIONS:")
# Most consistent high performers
config_groups = df.groupby(['n_blocks', 'n_channels', 'optimizer'])['avg_moves'].agg(['mean', 'count', 'std']).reset_index()
config_groups = config_groups[config_groups['count'] >= 5]  # At least 5 runs
config_groups = config_groups.sort_values('mean', ascending=False)
print("Top config combinations (with at least 5 runs):")
print(config_groups.head(10))

# 7. Parameter efficiency analysis
print("\n7. PARAMETER EFFICIENCY:")
df['moves_per_million_params'] = df['avg_moves'] / (df['model_parameters'] / 1000000)
efficiency = df.groupby(['n_blocks', 'n_hidden'])['moves_per_million_params'].agg(['mean', 'max']).round(2)
print("Efficiency (avg_moves per million parameters):")
print(efficiency.sort_values('mean', ascending=False))