from mpl_toolkits import mplot3d # necessary side effects
import torch
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm

matplotlib.use('WebAgg')

EPOCHS = 100000
LEARNING_RATE = 0.1

train_x = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]).reshape(-1, 2)
train_y = torch.tensor([0.0, 1.0, 1.0, 0.0]).reshape(-1, 1)

class NeuralXOR:
    def __init__(self):
        self.W_1 = torch.tensor([[0.37, -0.37], [0.37, -0.37]], requires_grad=True)
        self.W_2 = torch.tensor([[0.37], [0.37]], requires_grad=True)
        self.b_1 = torch.tensor([[-0.17]], requires_grad=True)
        self.b_2 = torch.tensor([[-0.17]], requires_grad=True)

    def _logits_h(self, x):
        return x @ self.W_1 + self.b_1

    def _logits_f(self, x):
        return self._h(x) @ self.W_2 + self.b_2

    def _h(self, x):
        return torch.sigmoid(self._logits_h(x))

    def f(self, x):
        return torch.sigmoid(self._logits_f(x))

    def loss_f(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self._logits_f(x), y)

model = NeuralXOR()

optimizer = torch.optim.SGD([model.W_1, model.b_1, model.W_2, model.b_2], LEARNING_RATE)
for epoch in tqdm(range(EPOCHS)):
    model.loss_f(train_x, train_y).backward()
    optimizer.step()

    optimizer.zero_grad()
    
for tensor in train_x:
    predicted_val = model.f(tensor)
    print('x_1=%s, x_2=%s, f(x_1, x_2)=%s' % (tensor[0], tensor[1], predicted_val))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('XOR operator')

ax.scatter3D(train_x[:, 0], train_x[:, 1], train_y, label='$x_1^{(i)}, x_2^{(i)}, y^{(i)}$')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$y$')

x1_train = train_x[:, 0]
x2_train = train_x[:, 1]

ax.scatter3D(x1_train, x2_train, train_y)

x1_axis = torch.linspace(torch.min(x1_train), torch.max(x1_train), 2)
x2_axis = torch.linspace(torch.min(x2_train), torch.max(x2_train), 2)

meshgrid = torch.meshgrid(x1_axis, x2_axis)
x = meshgrid[0]
y = meshgrid[1]

inputs = torch.stack((x.reshape(-1, 1), y.reshape(-1, 1)), 1).reshape(-1, 2)
z_axis = model.f(inputs).detach()
# print(z_axis)

ax_f = ax.plot_trisurf(
    x.reshape(1, -1)[0],
    y.reshape(1, -1)[0],
    z_axis.reshape(1, -1)[0],
    alpha=0.5,
    color='green'
)

plt.legend()
plt.show()