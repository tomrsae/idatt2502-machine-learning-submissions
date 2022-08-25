# This class is only meant to be used in this specific assignment.
# There is hardly any error handling etc. and is therefore not fit for any other use.

import torch

class CSVLoader:
    def load(filename):
        file = open(filename, 'r')
        file.readline() # Skip first line, as this is used to indicate column names

        vals = []
        for line in file:
            row = line.split(',')
            row[-1] = row[-1].replace('\n', '')
            row = ([float(x) for x in row])
            row = torch.tensor(row)
            #row = numpy.array(row).astype(numpy.float)

            vals.append(row)

        return torch.tensor(vals)