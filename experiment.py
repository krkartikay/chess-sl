# Experimentation framework

# Requirements:
#  - define a configuration for an experiment
#  - provide default values for the config params
#  - provide 'dev run' values for the params
#  - automatically run diff combinations of params to compare
#  - define functions to run for the experiments
#  - note down observations and store the values upon running the experiments

import time
import csv

from pathlib import Path

import observation


class Experiment:
    def __init__(self, start_function, dev_run=True, name=""):
        self.start_timestamp = time.time()
        self.start_function = start_function
        self.dev_run = dev_run
        self.possible_values = {}
        self.default_values = {}
        self.dev_run_values = {}
        self.observers = []
        self.selected_values = {}
        if len(name) > 0:
            self.name = name
        elif self.dev_run:
            self.name = "dev"
        else:
            self.name = f"{time.strftime('%Y%m%d_%I_%M_%S_%p')}"
        self.results_path = f"results/exp_{self.name}/"
        Path(self.results_path).mkdir(parents=True, exist_ok=True)
        self.run_number = 0
        self.all_configs_results = []

    def get_cont_observer(self, *args, **kwargs):
        observer = observation.ContinuousObserver(
            *args, path=self.results_path, suffix=f"{self.run_number:04d}", **kwargs)
        self.observers.append(observer)
        return observer

    def get_dist_observer(self, *args, **kwargs):
        observer = observation.DistributionObserver(
            *args, path=self.results_path, suffix=f"{self.run_number:04d}", **kwargs)
        self.observers.append(observer)
        return observer

    def add_config(self, name, possible_values,
                   default_value=None, dev_run_value=None):
        if default_value is None:
            default_value = possible_values[0]
        if dev_run_value is not None:
            self.dev_run_values[name] = dev_run_value
        self.possible_values[name] = possible_values
        self.default_values[name] = default_value

    def add_boolean_config(self, default_value=None, dev_run_value=None):
        self.add_config(self, [True, False],
                        dev_run_value=dev_run_value,
                        default_value=default_value)

    def get_config(self, name):
        if self.dev_run and name in self.dev_run_values:
            return self.dev_run_values[name]
        if name not in self.selected_values:
            return self.default_values[name]
        return self.selected_values[name]

    def save_results(self):
        csv_writer = csv.writer(
            open(self.results_path + "exp_results.csv", "w"))
        vars_report = list(self.all_configs_results[0].keys())
        csv_writer.writerow(vars_report)
        csv_writer.writerows([[results[var] for var in vars_report]
                             for results in self.all_configs_results])

    def run_experiment_full(self):
        # Main function!!!
        # We will run all possible combinations of variables and values.
        # Then we will run the start function for each different config and
        # collect all the observations. We will store all the observations
        # in the end.
        # We will also note down the time taken for each value of the variable
        # and note that down in a table and print it later.
        variables_to_vary = [
            x for (x, y) in self.possible_values.items()
            if (len(y) > 1 and not (self.dev_run and x in self.dev_run_values))]
        self.run_experiment_on_variables(variables_to_vary)

    def run_experiment_single_variable(self):
        # Main function!!!
        # We will select 1 variable to vary at a time and fix all the others
        # to their default values.
        # Then we will run the start function for each different config and
        # collect all the observations. We will store all the observations
        # in the end.
        # We will also note down the time taken for each value of the variable
        # and note that down in a table and print it later.
        variables_to_vary = [
            x for (x, y) in self.possible_values.items()
            if (len(y) > 1 and not (self.dev_run and x in self.dev_run_values))]
        for variable_to_select in variables_to_vary:
            for possible_value in self.possible_values[variable_to_select]:
                self.selected_values[variable_to_select] = possible_value
                self.run_experiment_with_selected_values()

    def run_experiment_on_variables(self, variables_list):
        # This function will select a value for each one variable turn by turn
        # and run the experiment when all variables are selected.

        # If all variables are selected then we can simply start a run
        if len(variables_list) == 0:
            self.run_experiment_with_selected_values()
            return

        # Let's take the first variable in the list and iterate over all the
        # values we can select
        variable_to_select = variables_list[0]
        new_variables_list = variables_list[1:]
        for possible_value in self.possible_values[variable_to_select]:
            self.selected_values[variable_to_select] = possible_value
            self.run_experiment_on_variables(variables_list=new_variables_list)

    def run_experiment_with_selected_values(self):
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
        results = self.start_function(self)
        end_time = time.time()
        time_taken = end_time - start_time
        print("===================================================")
        print(f"Run finished in {time_taken:3.2f}sec.")
        print("===================================================")
        print()
        self.all_configs_results.append({'run_num': self.run_number})
        for key, value in self.selected_values.items():
            self.all_configs_results[-1][key] = value
        for key, value in results.items():
            self.all_configs_results[-1][key] = value
        self.all_configs_results[-1]['running_time'] = time_taken
        for obs in self.observers:
            obs.plot()
            obs.write_csv()
            # obs.save_pkl()
        self.save_results()
