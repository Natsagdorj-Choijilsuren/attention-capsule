import torchvision
import torchvision.datasets as datasets
from torchvision import transforms

from torch.utils.data import DataLoader


def getMNIST_Loader(batch_size, shuffle=True):

    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(root = 'data/mnist', train=True,
                                   download=True, transform = transform)
    test_dataset = datasets.MNIST(root='data/mnist', train=False,
                                  download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False)

    return train_loader, test_loader



    

    
