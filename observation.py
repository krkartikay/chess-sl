# Observation framework

import os
import csv
import pickle
from matplotlib import pyplot as plt


class LogicalClock:
    def __init__(self):
        self.logical_time = 0

    def step(self):
        self.logical_time += 1


class Observer:
    def __init__(self, name, path="", suffix=""):
        self.name = name
        self.suffix = suffix
        self.fullname = os.path.join(
            path, name + (f"_{suffix}" if suffix else ""))
        self.observations = []

    def record(self, variable):
        self.observations.append(variable)

    def write_csv(self):
        filename = (f"{self.fullname}.csv")
        file_handle = open(filename, "a")
        csv_writer = csv.writer(file_handle)
        csv_writer.writerows([x] for x in self.observations)
        file_handle.close()

    def save_pkl(self):
        filename = (f"{self.fullname}.pkl")
        pickle.dump(self.observations, open(filename, "wb"))

    def load_pkl(self):
        filename = (f"{self.fullname}.pkl")
        return pickle.load(open(filename), "rb")


class ContinuousObserver(Observer):
    "Observe continuous-valued variables."

    def _plot(self):
        plt.plot(self.observations)

    def plot(self):
        filename = (f"{self.fullname}.png")
        plt.figure()
        self._plot()
        plt.savefig(filename)

    def avg(self):
        return sum(self.observations) / len(self.observations)

class DistributionObserver(Observer):
    "Observe random-valued variables."

    def _plot(self):
        plt.hist(self.observations)

    def plot(self):
        filename = (f"{self.fullname}.png")
        plt.figure()
        self._plot()
        plt.savefig(filename)

    def avg(self):
        return sum(self.observations) / len(self.observations)


def plot_together(observers):
    plt.figure()
    for obs in observers:
        obs._plot()
    plt.savefig("combined.png")
