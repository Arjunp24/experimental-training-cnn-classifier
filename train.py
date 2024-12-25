import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model_1 import Model_1
from model_2 import Model_2
from model_3 import Model_3
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm


# Training function
def train(epoch, train_loader, model, device, criterion, optimizer):
    model.train()
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        # Calculate accuracy
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        # Update running loss
        running_loss = 0.9 * running_loss + 0.1 * loss.item()
        pbar.set_postfix({
            'loss': f'{running_loss:.4f}',
            'acc': f'{100.0 * correct/total:.2f}%',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
    train_accuracy = 100. * correct / len(train_loader.dataset)
    print(f'Training Accuracy: {train_accuracy:.2f}%')


# Testing function
def test(model, test_loader, device, criterion):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return accuracy


# Main training loop
if __name__=="__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Define transforms with data augmentation
    train_transform = transforms.Compose([
        transforms.RandomRotation((-5, 5), fill=(1,)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load MNIST dataset
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=test_transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Initialize model, loss function, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Model_1().to(device)

# Print total number of trainable parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total trainable parameters: {total_params}')

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=2e-4)
scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.1,
    patience=2,
    verbose=True,
    min_lr=1e-5
)
print(f"Training on device: {device}")
print("Starting training for 15 epochs...")

for epoch in range(1, 16):
    _ = train(epoch, train_loader, model, device, criterion, optimizer)
    test_accuracy = test(model, test_loader, device, criterion)
    scheduler.step(test_accuracy)  # Update learning rate based on validation accuracy

print(f'Training completed. Final test accuracy: {test_accuracy:.2f}%')

model = Model_2().to(device)

# Print total number of trainable parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total trainable parameters: {total_params}')

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=2e-4)
scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.1,
    patience=2,
    verbose=True,
    min_lr=1e-5
)
print(f"Training on device: {device}")
print("Starting training for 15 epochs...")

for epoch in range(1, 16):
    _ = train(epoch, train_loader, model, device, criterion, optimizer)
    test_accuracy = test(model, test_loader, device, criterion)
    scheduler.step(test_accuracy)  # Update learning rate based on validation accuracy

print(f'Training completed. Final test accuracy: {test_accuracy:.2f}%')

model = Model_3().to(device)

# Print total number of trainable parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total trainable parameters: {total_params}')

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=2e-4)
scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.1,
    patience=2,
    verbose=True,
    min_lr=1e-5
)
print(f"Training on device: {device}")
print("Starting training for 15 epochs...")

for epoch in range(1, 16):
    _ = train(epoch, train_loader, model, device, criterion, optimizer)
    test_accuracy = test(model, test_loader, device, criterion)
    scheduler.step(test_accuracy)  # Update learning rate based on validation accuracy

print(f'Training completed. Final test accuracy: {test_accuracy:.2f}%')