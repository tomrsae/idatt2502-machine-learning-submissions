from mpl_toolkits import mplot3d # necessary side effects
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from csvloader import CSVLoader
from linearregressionmodel import LinearRegressionModel
from tqdm import tqdm

matplotlib.use('WebAgg')

EPOCHS = 10000
LEARNING_RATE = 0.0001
FILE_NAME = 'data/day_length_weight.csv'

values = CSVLoader.load_tensors_from(FILE_NAME)
day_data = values[0]
length_data = values[1]
weight_data = values[2]

x = torch.stack((length_data, weight_data), -1)

model = LinearRegressionModel(three_dim=True)

optimizer = torch.optim.SGD([model.W, model.b], LEARNING_RATE)
for epoch in tqdm(range(EPOCHS)):
    model.loss_f(x, day_data).backward()
    optimizer.step()

    optimizer.zero_grad()

print('W = %s, b = %s, loss = %s' % (model.W, model.b, model.loss_f(x, day_data)))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Length')
ax.set_ylabel('Weight')
ax.set_zlabel('Age (days)')

ax.scatter3D(length_data, weight_data, day_data)

x_axis = torch.linspace(torch.min(length_data), torch.max(length_data), 2)
y_axis = torch.linspace(torch.min(weight_data), torch.max(weight_data), 2)

x,y = torch.meshgrid(x_axis, y_axis)

inputs = torch.stack((x.reshape(-1, 1), y.reshape(-1, 1)), 1).reshape(-1, 2)

z = model.f(inputs).detach()

ax_f = ax.plot_trisurf(
    x.reshape(1, -1)[0],
    y.reshape(1, -1)[0],
    z.reshape(1, -1)[0],
    alpha=0.5,
    color='green'
)

plt.show()