# Observation framework

import os
import csv
import pickle
from collections.abc import Iterable
from matplotlib import pyplot as plt

class Observer:
    def __init__(self, name, path="", suffix=""):
        self.name = name
        self.suffix = suffix
        self.fullname = os.path.join(
            path, name + (f"_{suffix}" if suffix else ""))
        self.observations = []

    def record(self, variable):
        self.observations.append(variable)

    def write_csv(self, headings=None):
        filename = (f"{self.fullname}.csv")
        if len(self.observations) == 0:
            return
        if headings is None:
            headings = [self.name]
        with open(filename, "w") as file_handle:
            csv_writer = csv.writer(file_handle)
            csv_writer.writerow(headings)
            if isinstance(self.observations[0], Iterable):
                csv_writer.writerows(self.observations)
            else:
                csv_writer.writerows([x] for x in self.observations)

    def save_pkl(self):
        filename = (f"{self.fullname}.pkl")
        pickle.dump(self.observations, open(filename, "wb"))

    def plot(self, legend=None):
        if legend is None:
            legend = [self.name]
        filename = (f"{self.fullname}.png")
        plt.figure()
        plt.plot(self.observations)
        plt.legend(legend)
        plt.savefig(filename)

    def avg(self):
        return sum(self.observations) / len(self.observations)
