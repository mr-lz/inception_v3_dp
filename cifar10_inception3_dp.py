import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchcsprng as prng
from opacus import PrivacyEngine
from torchvision.datasets import CIFAR10
from torchvision import datasets, transforms
from inceptionv3 import Inception3
from tqdm import tqdm
N_RUNS = 1
SIGMA = 1
EPOCHS = 10
LR = 0.1
GRAD_NORM = 1
DELTA = 1e-5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_MODEL = False
DISABLE_DP = False
SECURE_RNG = False
DATA_ROOT = 'data/'

BATCH_SIZE = 4
normalize = [
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
]
transform = transforms.Compose(normalize)


train_dataset = CIFAR10(
    DATA_ROOT, train=True, download=True, transform=transform
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
)

test_dataset = CIFAR10(
    root=DATA_ROOT, train=False, download=True, transform=transform
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []
    for _batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
#         print(data.requires_grad)
        optimizer.zero_grad()
        output, _ = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    if not DISABLE_DP:
        epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(DELTA)
        print(
            f"Train Epoch: {epoch} \t"
            f"Loss: {np.mean(losses):.6f} "
            f"(ε = {epsilon:.2f}, δ = {DELTA}) for α = {best_alpha}"
        )
    else:
        print(f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f}")


def test(model, device, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    return correct / len(test_loader.dataset)


def main():
    run_results = []
    for _ in range(N_RUNS):
        model = Inception3(num_classes=10).to(DEVICE)

        optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0)
        if not DISABLE_DP:
            privacy_engine = PrivacyEngine(
                model,
                batch_size=BATCH_SIZE,
                sample_size=len(train_loader.dataset),
                alphas=[
                    1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
                noise_multiplier=SIGMA,
                max_grad_norm=GRAD_NORM,
                secure_rng=SECURE_RNG,
            )
            privacy_engine.attach(optimizer)
        for epoch in range(1, EPOCHS + 1):
            train(model, DEVICE, test_loader, optimizer, epoch)
        run_results.append(test(model, DEVICE, test_loader))

    if len(run_results) > 1:
        print(
            "Accuracy averaged over {} runs: {:.2f}% ± {:.2f}%".format(
                len(run_results), np.mean(run_results) *
                100, np.std(run_results) * 100
            )
        )


if __name__ == "__main__":
    main()
