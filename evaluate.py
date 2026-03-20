import torch
from model import MMANet
from preprocessing import get_dataloaders

model = MMANet()
model.load_state_dict(torch.load("mma_net_checkpoint.pth"))
model.eval()

_, test_loader = get_dataloaders()

correct = 0
total = 0

with torch.no_grad():
    for batch in test_loader:
        voice = batch['voice']
        gait = batch['gait']
        labels = batch['label'].squeeze()

        outputs = model(voice, gait)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
