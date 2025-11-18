import torch

# 2)

tensor_a = torch.rand(size=(7,7))
#print(tensor_a)

# 3)

tensor_b = torch.rand(size=(1,7))
#print(tensor_b)
#print(torch.matmul(tensor_a, tensor_b.T))

# 4)

RANDOM_SEED = 0

torch.manual_seed(seed = RANDOM_SEED)
tensor_a = torch.rand(size=(7,7))

torch.manual_seed(seed = RANDOM_SEED)
tensor_b = torch.rand(size=(1,7))

#print(torch.matmul(tensor_a,tensor_b.T))

# 5)

RANDOM_SEED_CUDA = 1234
torch.cuda.manual_seed(seed = RANDOM_SEED_CUDA)

# 6)

RANDOM_SEED_2 = 1234

torch.manual_seed(seed = RANDOM_SEED_2)
tensor_c = torch.rand(size=(2, 3))
tensor_c = tensor_c.to(device='cuda')

torch.manual_seed(seed = RANDOM_SEED_2)
tensor_d = torch.rand(size=(2, 3))
tensor_d = tensor_d.to(device='cuda')

# 7)
torch_mul = torch.matmul(tensor_c, tensor_d.T)

# 8)

print(f'Minimum: {torch_mul.min()}')
print(f'Maximum: {torch_mul.max()}')

# 9)

print(f'Minimum index: {torch_mul.argmin()}')
print(f'Maximum index: {torch_mul.argmax()}')

# 10)

torch.manual_seed(7)
torch_e = torch.rand(1, 1, 1, 10)

torch_f = torch_e.squeeze()

print(f'First tensor and it\'s shape {torch_e, torch_e.shape}')
print(f'Second tensor and it\'s shape {torch_f, torch_f.shape}')
