import torch

# CREATE TENSORS
e = torch.empty(2, 3)                        # empty tensor
e1 = torch.rand(2, 2, dtype=torch.float16)   # tensor with random values
# pot adauga  "dtype" ca sa decid tipul datelor din tensor
e2 = torch.ones(2, 2)                       # tensor with ones
e3 = torch.zeros(2, 3)                      # tensor with zeros
e3 = torch.tensor([0.2, 0.3, 0.4])


# tensor operations
x = torch.rand(2, 2)
y = torch.rand(2, 2)

# methods to do operations
z = x + y
z = torch.add(x, y)
y.add_(x)  # in place addition


# print(x[1,1])        #prints the tensor
# print(x[1,1].item()) #prints only the value


# reshaping
x = torch.rand(4, 4)
y = x.view(16)
z = x.view(-1, 8)      # pytorch will automatically determine the right size for the -1


# converting numpy to tensors and viceversa

a = torch.rand(2)             # -->tensor
b = a.numpy()                 # -->numpy array
a = torch.from_numpy(b)       # -->tensor

print(a, b)
print(torch.cuda.is_available())

if torch.cuda.is_available():

    print("Hello1")

    device = torch.device("cuda")

    # create a torch on the device
    x = torch.ones(4, device=device)

    # create a torch and send it to the device
    y = torch.ones(4)
    y.to(device)

    # z=y.numpy() # error cause numpy can only convert CPU tensors

print("hello2")
