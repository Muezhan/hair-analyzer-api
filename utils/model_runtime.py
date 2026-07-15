import torch
print(f"Running PyTorch Version         : {torch.__version__}")
print(f"CUDA Version                    :{torch.version.cuda}")
print(f"CUDA Available                  :{torch.cuda.is_available()}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Using device: {device.type.upper()} ---")

def is_cuda():
    return device.type.upper() == 'CUDA'