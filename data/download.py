import torchvision
import torchvision.transforms as transforms

# Transformation (convert to tensor, normalize)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                         (0.2470, 0.2435, 0.2616))
])

# Download CIFAR-10
trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform)

print(f"Train size: {len(trainset)} | Test size: {len(testset)}")
