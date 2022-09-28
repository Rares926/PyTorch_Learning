import torch
import torch.nn as nn


X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

X_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = X.shape  # will be 4,1


input_size = n_features
output_size = n_features


class LinearRegression(nn.Module):

    def __init__(self, input_dim, output_dim) -> None:
        super(LinearRegression, self).__init__()
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)


# model=nn.Linear(input_size,output_size)
model = LinearRegression(input_size, output_size)

# trebuie pus item ca sa scoata valoarea ca daca e tensor da eroare
print(f'Prediction before training f(5)={model(X_test).item():.3f}')

# Training

learning_rate = 0.01
n_iters = 100
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = model(X)

    # loss
    l = loss(Y, y_pred)    # noqa: E741

    # compute gradients
    l.backward()  # dl/dw

    # update weights
    optimizer.step()

    # zero gradients after weights update
    optimizer.zero_grad()

    # print some information
    if epoch % 10 == 0:
        [w, b] = model.parameters()
        print(f'epoch{epoch+1}: w={w[0][0].item():.3f}, loss {l:.8f}')

print(f'Prediction after training f(5)={model(X_test).item():.3f}')
