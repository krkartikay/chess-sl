import wandb
import pandas as pd
import sys

sweep_id = sys.argv[1] if len(sys.argv) > 1 else None
if not sweep_id:
    print("Usage: python get_sweep_data.py <sweep_id>")
    sys.exit(1)
api = wandb.Api()
runs = api.sweep(sweep_id).runs
pd.DataFrame([{**run.config, **run.summary._json_dict, 'name': run.name, 'state': run.state} for run in runs]).to_csv('sweep_results.csv', index=False)