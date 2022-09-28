import numpy as np 

# f = w * x
# if w is equal to 2 this will become
# f = 2 * x

X = np.array([1, 2, 3, 4], dtype=np.float32)
Y = np.array([2, 4, 6, 8], dtype=np.float32)

w = 0.0

# model prediction


def forward(x):
    return w*x


# loss
def loss(y, y_prediction):
    return ((y_prediction-y)**2).mean()

# gradient
# MSE =1/N *(w*x-y)**2
# dJ/dw=1/N 2x (w*x -y )


def gradient(x, y, y_predicted):
    return np.dot(2*x, y_predicted-y)/x.size


print(f'Prediction before training f(5)={forward(5):.3f}')

# Training
learning_rate = 0.01
n_iters = 20

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)  # noqa: E741

    # gradients
    dw = gradient(X, Y, y_pred)

    # update weights
    w -= learning_rate*dw

    # print some information
    if epoch % 2 == 0:
        print(f'epoch{epoch+1}: w={w:.3f}, loss {l:.8f}')

print(f'Prediction after training f(5)={forward(5):.3f}')
