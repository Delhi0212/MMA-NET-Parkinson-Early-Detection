#Import
import torch
from model import *
from preprocessing import get_dataloaders
#Load model
model = MMANet()   # or your model class name
model.load_state_dict(torch.load("mma_net_checkpoint.pth"))
model.eval()
#Load test data
_, test_loader = get_dataloaders()
correct = 0
total = 0
#Evaluation
with torch.no_grad():
    for batch in test_loader:
        voice = batch['voice']
        gait = batch['gait']
        labels = batch['label']

        outputs = model(voice, gait)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()
#Print
accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
