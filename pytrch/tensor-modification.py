import torch
# view : Reshapes a tensor to a new shape without changing its data.
print("Demonstration of view\n\n")
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(x)
print(x.shape)   # (2, 3)
x = x.view(3, 2)
print(x)
print(x.shape)  # reshape to (3,2)
print("done\n\n")

# squeeze : Remove all single dimensions from a tensor
print("Demonstration of squeeze\n\n")
x = torch.tensor([[[1, 3, 4], [5, 6, 7]]])
print(x)
print(x.shape)   # (1, 2, 3)
x = x.squeeze()
print(x)
print(x.shape)   # (2, 3)
print("done\n\n")

# unsqueeze : Insert a dimension of size one into a tensor
print("Demonstration of unsqueeze\n\n")
x = torch.tensor([[1, 3, 4], [5, 6, 7]])
print(x)
print(x.shape)   # (2, 3)
x0 = x.unsqueeze(0)
x1 = x.unsqueeze(1)
print(x0)
print(x0.shape)   # (1, 2, 3)
print("\n")
print(x1)
print(x1.shape)   # (2, 1, 3)
print("done\n\n")


# permute : Rearranges the dimensions of a tensor
print("Demonstration of permute\n\n")
x = torch.tensor([[1, 3, 4], [5, 6, 7]])
print(x)
print(x.shape)   # (2, 3)
x = x.permute(1, 0)
print(x)
print(x.shape)   # (3, 2)
print("done\n\n\n")

# transpose : Swaps the dimensions of a tensor
print("Demonstration of transpose\n\n")
x = torch.tensor([[1, 3, 4], [5, 6, 7]])
print(x)
print(x.shape)   # (2, 3)
x = x.transpose(0, 1)
print(x)
print(x.shape)   # (3, 2)
print("done\n\n\n")

# flatten : Return a copy of the tensor collapsed into one dimension
print("Demonstration of flatten\n\n")
x = torch.tensor([[1, 3, 4], [5, 6, 7]])
print(x)
print(x.shape)   # (2, 3)
x = x.flatten()
print(x)
print(x.shape)   # (6)
print("done\n\n\n")

# repeat : Repeat this tensor along specified dimensions
print("Demonstration of repeat\n\n")
x = torch.tensor([[1, 3, 4], [5, 6, 7]])
print(x)
print(x.shape)   # (2, 3)
x = x.repeat(2, 2)
print(x)
print(x.shape)   # (4, 6)
print("done\n\n\n")

# cat : Concatenates the given sequence of tensors along a specified dimension
print("Demonstration of cat\n\n")
x = torch.tensor([[1, 3, 4], [5, 6, 7]])
print(x)
print(x.shape)   # (2, 3)
x = torch.cat([x, x], dim=0)
print(x)
print(x.shape)   # (4, 3)
print("done\n\n\n")

# stack : Stacks tensors along a new dimension
print("Demonstration of stack\n\n")
x = torch.tensor([[1, 3, 4], [5, 6, 7]])
print(x)
print(x.shape)   # (2, 3)
x = torch.stack([x, x], dim=0)
print(x)
print(x.shape)   # (2, 2, 3)
print("done\n\n\n")

# unbind : Unpacks the tensor into a tuple of tensors along a specified dimension
print("Demonstration of unbind\n\n")
x = torch.tensor([[1, 3, 4], [5, 6, 7]])
print(x)
print(x.shape)   # (2, 3)
x = torch.unbind(x, dim=0)
print(x)
print(len(x))   # 2
print("done\n\n\n")

# split : Splits the tensor into a list of tensors along a specified dimension
print("Demonstration of split\n\n")
x = torch.tensor([[1, 3, 4], [5, 6, 7]])
print(x)
print(x.shape)   # (2, 3)
x = torch.split(x, 2, dim=0)
print(x)
print(len(x))   # 2
print("done\n\n\n")

# chunk : Splits the tensor into a list of tensors along a specified dimension
print("Demonstration of chunk\n\n")
x = torch.tensor([[1, 3, 4], [5, 6, 7]])
print(x)
print(x.shape)   # (2, 3)
x = torch.chunk(x, 2, dim=0)
print(x)
print(len(x))   # 2
print("done\n\n\n")
