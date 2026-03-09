import sys
import torch
import datasets
import streamlit

print("Python:", sys.version)
print("Torch:", torch.__version__)
print("Datasets:", datasets.__version__)
print("Streamlit:", streamlit.__version__)
print("CUDA available:", torch.cuda.is_available())
print("MPS available:", torch.backends.mps.is_available())
