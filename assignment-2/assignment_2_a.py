import torch
import matplotlib.pyplot as plt
# import matplotlib
from tqdm import tqdm

# matplotlib.use('WebAgg')

EPOCHS = 1000
LEARNING_RATE = 10

train_x = torch.tensor([0.0, 1.0]).reshape(-1, 1)
train_y = torch.tensor([1.0, 0.0]).reshape(-1, 1)

class NeuralNOT:
    def __init__(self):
        self.W = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    def _logits(self, x):
        return x @ self.W + self.b

    def f(self, x):
        return torch.sigmoid(self._logits(x))

    def loss_f(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self._logits(x), y)

model = NeuralNOT()

optimizer = torch.optim.SGD([model.W, model.b], LEARNING_RATE)
for epoch in tqdm(range(EPOCHS)):
    model.loss_f(train_x, train_y).backward()
    optimizer.step()

    optimizer.zero_grad()

print('W = %s, b = %s, loss = %s' % (model.W, model.b, model.loss_f(train_x, train_y)))

plt.plot(train_x, train_y, 'o', label='$(x^{(i)},y^{(i)})$')
plt.xlabel('x')
plt.ylabel('y')

x = torch.tensor([[torch.min(train_x)], [torch.max(train_y)]])
plt.plot(x, model.f(x).detach(), label='$\\hat y = f(x) = \\sigma(xW+b$)')

print(model.f(torch.tensor([0.0])))
print(model.f(torch.tensor([1.0])))

plt.legend()
plt.show()