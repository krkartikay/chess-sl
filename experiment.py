# Experimentation framework

import csv
import itertools
import time
import pathlib
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Any
from observer import ALL_OBSERVERS


@dataclass
class Config:
    name: str
    values: List[Any] = field(default_factory=lambda: [])
    default: Any = None
    dev: Any = None
    expt: Any = None

    def __post_init__(self):
        ALL_CONFIGS.append(self)
        if self.default is None:
            self.default = self.values[0]
        if self.dev is None:
            self.dev = self.default
        if self.default not in self.values:
            self.values.append(self.default)

    def get(self):
        if self.expt is None:
            return self.dev
        return self.expt.get_config(self)


ALL_CONFIGS: List[Config] = []


class Experiment:
    def __init__(self, variables=[], dev_mode=False):
        self.variables: List[Config] = variables
        self.dev_mode: bool = dev_mode

        self.start_time = time.time()
        if self.dev_mode:
            self.name = "dev"
        else:
            self.name = f"{time.strftime('%Y_%m_%d__%H_%M_%S')}"
        self.results_path = f"results/exp_{self.name}/"
        pathlib.Path(self.results_path).mkdir(parents=True, exist_ok=True)

        self.run_number: int = 0
        self.selected_values: Dict[str, Any] = {}
        self.all_configs_results: List[Dict[str, Any]] = []
        for config in ALL_CONFIGS:
            config.expt = self

    def get_config(self, config):
        if config.name in self.selected_values:
            return self.selected_values[config.name]
        elif self.dev_mode:
            return config.dev
        else:
            return config.default

    def run_experiment(self,
                       function=None,
                       time_limit: int = 0):
        value_combinations = itertools.product(
            *[v.values for v in self.variables])
        for values in value_combinations:
            self.run_experiment_with_values(function, values)

    def run_experiment_with_values(self, function, values):
        self.selected_values = {
            self.variables[i].name: values[i]
            for i in range(len(self.variables))}
        # run_experiment will run this function with selected values of
        # variables
        self.run_number += 1
        print()
        print("===================================================")
        print(f"Experiment run {self.run_number}")
        print("---------------------------------------------------")
        print(f"Config for this run:\n{self.selected_values}")
        print("===================================================")
        start_time = time.time()
        results, model = function()
        end_time = time.time()
        time_taken = end_time - start_time
        print("===================================================")
        print(f"Run finished in {time_taken:3.2f}sec.")
        print("===================================================")
        print()
        self.all_configs_results.append({'run_num': self.run_number})
        for config in ALL_CONFIGS:
            self.all_configs_results[-1][config.name] = config.get()
        for key, value in results.items():
            self.all_configs_results[-1][key] = value
        self.all_configs_results[-1]['running_time'] = time_taken
        for obs in ALL_OBSERVERS:
            obs.set_path(path=self.results_path,
                         suffix=f"{self.run_number:03d}")
            obs.plot()
            obs.write_csv()
        # Save the model and the results
        if model is not None:
            with open(self.results_path 
                      + f"model_{self.run_number:03d}.pt", "wb") as model_file:
                torch.save(model.state_dict(), model_file)
        self.save_results()

    def save_results(self):
        csv_writer = csv.writer(
            open(self.results_path + "exp_results.csv", "w"))
        vars_report = list(self.all_configs_results[0].keys())
        csv_writer.writerow(vars_report)
        csv_writer.writerows([[results[var] for var in vars_report]
                              for results in self.all_configs_results])
