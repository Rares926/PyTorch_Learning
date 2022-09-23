import torch


# x=torch.rand(3,requires_grad=True)
# # print(x)

# y=x+2
# # print(y)

# z=y*y*2
# # z=z.mean()
# # print(z)

# v=torch.rand(3,dtype=torch.float32)
# z.backward(v)  # this will calculate the gradients of z with respect to x
# # print(x.grad)


x=torch.rand(3,requires_grad=True)                          #create a tensor with grad
# x=tensor([0.8243, 0.6635, 0.9086], requires_grad=True)
# x.requires_grad_(False)                                     #elimine the grad inside the tensor
# x=tensor([0.8243, 0.6635, 0.9086])
# y=x.detach()                                                #create a copy of the tensor that has grads without the grads
# y=tensor([0.7664, 0.7230, 0.4522])


# with torch.no_grad(): #the gradients in x will not be used so y will not have the gradient function
#     y=x+2
#     print(y)


#   DUMMY TRAINING EXAMPLE

weights=torch.ones(4,requires_grad=True)

for epoch in range(3):
    model_output=(weights*3).sum()


    #compute gradients
    model_output.backward() #this will give us the gradients 

    print(weights.grad)

    #we must empy the gradients before the next optimization step 
    #verry important during training 
    weights.grad.zero_()

