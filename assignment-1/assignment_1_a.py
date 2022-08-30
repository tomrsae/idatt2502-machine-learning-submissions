import torch
import matplotlib.pyplot as plt
import matplotlib
from csvloader import CSVLoader
from linearregressionmodel import LinearRegressionModel
from tqdm import tqdm

matplotlib.use('WebAgg')

EPOCHS = 1000000
LEARNING_RATE = 0.0001 
FILE_NAME = 'data/length_weight.csv'

values = CSVLoader.load_tensors_from(FILE_NAME)
length_data = values[0]
weight_data = values[1]

model = LinearRegressionModel()

optimizer = torch.optim.SGD([model.W, model.b], LEARNING_RATE)
for epoch in tqdm(range(EPOCHS)):
    model.loss_f(length_data, weight_data).backward()
    optimizer.step()

    optimizer.zero_grad()

print('W = %s, b = %s, loss = %s' % (model.W, model.b, model.loss_f(length_data, weight_data)))

plt.plot(length_data, weight_data, 'o', label='$(x^{(i)},y^{(i)})$')
plt.xlabel('Length')
plt.ylabel('Weight')

x = torch.tensor([[torch.min(length_data)], [torch.max(length_data)]])
plt.plot(x, model.f(x).detach(), label='$\\hat y = f(x) = xW+b$')

plt.legend()
plt.show()