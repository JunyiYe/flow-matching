import argparse
import torch
import matplotlib.pyplot as plt
from config import DEVICE, MODEL_PATH
from models.unet import UNet

def generate_image(model, label, steps=250):
    x = torch.randn(1, 1, 28, 28).to(DEVICE)
    label = torch.tensor([label], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        for i in range(steps):
            t = torch.tensor([1.0 / steps * i], device=DEVICE)
            pred_vt = model(x, t, label)
            x = x + pred_vt * (1.0 / steps)
            x = x.detach()

    x = (x + 1) / 2
    return x[0, 0].cpu().numpy()

def visualize_image(img_array):
    plt.figure(figsize=(1, 1))
    plt.axis("off")
    plt.imshow(img_array, cmap="gray")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', type=int, default=5, help='Digit label to condition on (0-9)')
    parser.add_argument('--steps', type=int, default=250, help='Number of denoising steps')
    args = parser.parse_args()

    model = UNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    img = generate_image(model, label=args.label, steps=args.steps)
    visualize_image(img)
