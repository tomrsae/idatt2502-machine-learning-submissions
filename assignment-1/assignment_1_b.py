from mpl_toolkits import mplot3d # necessary side effects
import torch
import numpy as np
import matplotlib.pyplot as plt
from csvloader import CSVLoader
from linearregressionmodel import LinearRegressionModel

EPOCHS = 1000000
LEARNING_RATE = 0.0001
FILE_NAME = 'data/day_length_weight.csv'

values = CSVLoader.load_tensors_from(FILE_NAME)
day_data = values[0]
length_data = values[1]
weight_data = values[2]

x = torch.stack((length_data, weight_data), -1)

model = LinearRegressionModel(three_dim=True)

optimizer = torch.optim.SGD([model.W, model.b], LEARNING_RATE)
for epoch in range(EPOCHS):
    model.loss(x, day_data).backward()
    optimizer.step()

    optimizer.zero_grad()

print('W = %s, b = %s, loss = %s' % (model.W, model.b, model.loss(x, day_data)))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Length')
ax.set_ylabel('Weight')
ax.set_zlabel('Day')

ax.scatter3D(length_data, weight_data, day_data)

x_axis = np.linspace(torch.min(length_data), torch.max(length_data), 2)
y_axis = np.linspace(torch.min(weight_data), torch.max(weight_data), 2)

x1_grid, x2_grid = np.meshgrid(x_axis, y_axis)
y_grid = torch.tensor([
    [torch.min(length_data), torch.min(weight_data)], 
    [torch.max(length_data), torch.max(weight_data)]
])

ax_f = ax.plot_surface(x1_grid, x2_grid, model.f(y_grid).detach(), color='green')

plt.show()