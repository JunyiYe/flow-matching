import argparse
import torch
from tqdm import tqdm
import os

from config import DEVICE, MODEL_PATH
from models.unet import UNet
from datasets.mnist_loader import get_mnist_loader


def train_model(epochs, lr, batch_size):
    model = UNet().to(DEVICE)
    dataloader = get_mnist_loader(batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        losses = []
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for img, labels in pbar:
            img, labels = img.to(DEVICE), labels.to(DEVICE)
            t = torch.rand(size=(img.size(0),), device=DEVICE)
            noise = torch.randn_like(img)
            xt = (1 - t.view(-1, 1, 1, 1)) * noise + t.view(-1, 1, 1, 1) * img

            pred_vt = model(xt, t, labels)
            loss = torch.nn.functional.mse_loss(pred_vt, img - noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            pbar.set_postfix(loss=loss.item())

        torch.save(model.state_dict(), f".model.pt")
        os.replace(".model.pt", MODEL_PATH)
        avg_loss = sum(losses) / len(losses)
        print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.6f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    args = parser.parse_args()

    train_model(args.epochs, args.lr, args.batch_size)
