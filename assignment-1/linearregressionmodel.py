# Class is retrieved from https://gitlab.com/ntnu-tdat3025/regression/linear-2d/-/blob/master/main.py
# This code is authored by lecturer Ole Christian Eidheim, with modifications made by me. 

import torch

class LinearRegressionModel:

    def __init__(self, three_dim=False):
        # Model variables
        if three_dim:
            self.W = torch.tensor([[0.0], [0.0]], requires_grad=True)
        else:
            self.W = torch.tensor([[0.0]], requires_grad=True)  # requires_grad enables calculation of gradients
        self.b = torch.tensor([[0.0]], requires_grad=True)

    # Predictor
    def f(self, x):
        return x @ self.W + self.b  # @ corresponds to matrix multiplication

    def g(self, x):
        return 20 * torch.sigmoid(self.f(x)) + 31
        #return 20 * torch.nn.functional.sigmoid(self.f(x).detach()) + 31

    # Uses Mean Squared Error
    def loss_f(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))  # Can also use torch.nn.functional.mse_loss(self.f(x), y) to possibly increase numberical stability

    def loss_g(self, x, y):
        return torch.mean(torch.square(self.g(x) - y))
