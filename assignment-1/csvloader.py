# This class is only meant to be used in this specific assignment.
# There is no error handling etc. and is therefore likely not fit for any other use.

import torch

class CSVLoader:
    def load_tensors_from(filename):
        file = open(filename, 'r')
        columns = len(file.readline().split(','))

        tensors = [[] for i in range(columns)]
        for line in file:
            row = line.split(',')
            row[-1] = row[-1].replace('\n', '')

            for i, val in enumerate(row):
                tensors[i].append(float(val))

        axis_vals = []
        for tensor in tensors:
            axis_vals.append(torch.tensor(tensor).reshape(-1, 1))

        return axis_vals