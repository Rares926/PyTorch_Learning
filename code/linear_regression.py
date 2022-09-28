import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


# 0)prepare data

X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))

y = y.view(y.shape[0], 1)

n_samples, n_features = X.shape


# 1)define the model

input_size = n_features
output_size = 1


class LinearRegression(nn.Module):

    def __init__(self, input_dim, output_dim) -> None:
        super(LinearRegression, self).__init__()
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)


# model=nn.Linear(input_size,output_size)
model = LinearRegression(input_size, output_size)

# 2)define the loss and optimizer

criterion = nn.MSELoss()
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3)training loop

num_epoch = 100

for epoch in range(num_epoch):

    # forward pass and loss
    y_predicted = model(X)
    loss = criterion(y_predicted, y)

    # bacward pass
    loss.backward()

    # update
    optimizer.step()

    # empty the gradients 
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f'epoch {epoch+1}, loss={loss.item():.4f}')


# plot
predicted = model(X).detach().numpy()  # detach so the values of the gradients not be saved

plt.plot(X_numpy, y_numpy, "ro")
plt.plot(X_numpy, predicted, 'b')

plt.show()
