import os
print(os.getcwd())
print(os.path.exists("data/faces"))
print(os.path.exists("./data/faces"))
print(os.path.exists(r"data\faces"))