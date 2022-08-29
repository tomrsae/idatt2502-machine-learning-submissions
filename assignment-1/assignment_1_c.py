from requests import head
import torch
import matplotlib.pyplot as plt
from csvloader import CSVLoader
from linearregressionmodel import LinearRegressionModel

EPOCHS = 1000000
LEARNING_RATE = 0.000001
FILE_NAME = 'data/day_head_circumference.csv'

values = CSVLoader.load_tensors_from(FILE_NAME)
day_data = values[0]
head_circumference_data = values[1]

model = LinearRegressionModel()

optimizer = torch.optim.SGD([model.W, model.b], LEARNING_RATE)
for epoch in range(EPOCHS):
    model.loss_g(day_data, head_circumference_data).backward()
    optimizer.step()

    optimizer.zero_grad()

print('W = %s, b = %s, loss = %s' % (model.W, model.b, model.loss_g(day_data, head_circumference_data)))

plt.plot(day_data, head_circumference_data, 'o', label='$(x^{(i)},y^{(i)})$')
plt.xlabel('Age (days)')
plt.ylabel('Head circumference')

x = torch.tensor([[torch.min(day_data)], [torch.max(day_data)]])
plt.plot(x, model.g(x).detach(), label='$\\hat y = f(x) = 20\\sigma(xW+b) + 31$')

plt.legend()
plt.show()