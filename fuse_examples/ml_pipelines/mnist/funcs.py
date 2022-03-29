import torchvision
from torchvision import transforms

def run_train():
    pass

def run_eval():
    pass

def run_infer():
    pass

def create_dataset(params):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    torch_train_dataset = torchvision.datasets.MNIST(params['common']['cache_dir'], download=True, train=True, transform=transform)
    return torch_train_dataset
