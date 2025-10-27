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