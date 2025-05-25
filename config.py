import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EPOCHS = 10
MODEL_PATH = "model.pt"
