from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_mnist_loader(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 2 * x - 1)
    ])
    dataset = datasets.MNIST(root='data/mnist_data', train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)