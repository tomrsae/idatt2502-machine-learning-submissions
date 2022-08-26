import torch
import matplotlib.pyplot as plt
from csvloader import CSVLoader
from linearregressionmodel import LinearRegressionModel

EPOCHS = 1000000
LEARNING_RATE = 0.0001 
FILE_NAME = 'data/length_weight.csv'

values = CSVLoader.load_tensors_from(FILE_NAME)

model = LinearRegressionModel()

optimizer = torch.optim.SGD([model.W, model.b], LEARNING_RATE)
for epoch in range(EPOCHS):
    model.loss(values[0], values[1]).backward()
    optimizer.step()

    optimizer.zero_grad()

print('W = %s, b = %s, loss = %s' % (model.W, model.b, model.loss(values[0], values[1])))

plt.plot(values[0], values[1], 'o', label='$(x^{(i)},y^{(i)})$')
plt.xlabel('Length')
plt.ylabel('Weight')

x = torch.tensor([[torch.min(values[0])], [torch.max(values[0])]])
plt.plot(x, model.f(x).detach(), label='$\\hat y = f(x) = xW+b$')

plt.legend()
plt.show()