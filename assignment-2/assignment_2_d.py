import torch
import torchvision
import matplotlib.pyplot as plt
# import matplotlib
from tqdm import tqdm

# matplotlib.use('WebAgg')

EPOCHS = 100000
LEARNING_RATE = 0.1
DATASET_SIZE = 784

mnist_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize([0.], [1.])])
mnist_train = torchvision.datasets.MNIST('./data', train=True, download=True, transform=mnist_transform)
train_x = mnist_train.data.reshape(-1, DATASET_SIZE).float()
train_y = torch.zeros((mnist_train.targets.shape[0], 10))
train_y[torch.arange(mnist_train.targets.shape[0]), mnist_train.targets] = 1

print(train_x[0])

class NeuralMNIST:
    def __init__(self):
        self.W = torch.zeros((DATASET_SIZE, 28), requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    def logits(self, x):
        return x @ self.W + self.b

    def f(self, x):
        return torch.softmax(self.logits(x))

    def loss(self, x, y):
        return torch.nn.functional.cross_entropy(self.logits(x), y)

    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())

model = NeuralMNIST()

print('Optimizing model')
optimizer = torch.optim.SGD([model.W, model.b], LEARNING_RATE)
for epoch in tqdm(range(EPOCHS)):
    model.loss(train_x, train_y).backward()
    optimizer.step()

    optimizer.zero_grad()

mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True)
test_x = mnist_test.data.reshape(-1, 784).float()
test_y = torch.zeros((mnist_test.targets.shape[0], 10))
test_y[torch.arange(mnist_test.targets.shape[0]), mnist_test.targets] = 1

plt.imshow(train_x[0, :].reshape(28, 28))

print(train_y[0, :])

plt.show()