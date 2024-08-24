import os
import time
import torch
import torchvision
import timm

from torchvision import transforms
from tqdm import tqdm

import torch.optim as optim
import torch.nn as nn
from PIL import Image

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, random_split

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {} 
        self.classes = []

        self._load_dataset()

    def _load_dataset(self):
        for class_idx, class_dir in enumerate(sorted(os.listdir(self.root_dir))):
            class_path = os.path.join(self.root_dir, class_dir)
            if os.path.isdir(class_path):
                self.class_to_idx[class_dir] = class_idx  
                self.classes.append(class_dir)  
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                        self.image_paths.append(img_path)
                        self.labels.append(class_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

os.environ["MASTER_ADDR"] = "200.17.78.37" 
os.environ["MASTER_PORT"] = "4500" 
os.environ["WORLD_SIZE"] = "2" 
os.environ["RANK"] = "0"  
os.environ["LOCAL_RANK"] = "0"  

dist.init_process_group("nccl")

local_rank = int(os.environ["LOCAL_RANK"])
global_rank = int(os.environ["RANK"])

BATCH_SIZE = 32 // int(os.environ["WORLD_SIZE"])
EPOCHS = 5
WORKERS = 48
IMG_DIMS = (256, 256)
CLASSES = 10

MODEL_NAME = 'resnet34'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(IMG_DIMS),
])

dataset = CustomImageDataset(root_dir="dataset/train_images", transform=transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_sampler = DistributedSampler(train_dataset)
test_sampler = DistributedSampler(test_dataset)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=False,
                                           sampler=train_sampler,
                                           num_workers=WORKERS)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=BATCH_SIZE,
                                          shuffle=False,
                                          sampler=test_sampler,
                                          num_workers=WORKERS)

torch.cuda.set_device(local_rank)
torch.cuda.empty_cache()
model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=CLASSES)

model = model.to('cuda:' + str(local_rank))
model = DDP(model, device_ids=[local_rank])

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

start = time.perf_counter()
for epoch in range(EPOCHS):
    epoch_start_time = time.perf_counter()
    
    model.train()
    train_sampler.set_epoch(epoch)

    running_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(train_loader, total=len(train_loader)):
        features, labels = batch[0].to(local_rank), batch[1].to(local_rank)
        print("\n\n LABELS", labels)

        optimizer.zero_grad()

        preds = model(features)
        loss = loss_fn(preds, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(preds, 1)
        print("PREDICTED", predicted)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        print("running_loss", running_loss)
        print("running_accuracy", correct/total)

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total

    epoch_end_time = time.perf_counter()
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Loss: {epoch_loss:.4f} | Accuracy: {epoch_accuracy:.2f}%")
    print(f"Epoch Time: {epoch_end_time - epoch_start_time:.2f}s")

model.eval()
test_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for batch in tqdm(test_loader, total=len(test_loader)):
        features, labels = batch[0].to(local_rank), batch[1].to(local_rank)

        preds = model(features)
        loss = loss_fn(preds, labels)
        test_loss += loss.item()

        _, predicted = torch.max(preds, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

average_test_loss = test_loss / len(test_loader)
accuracy = 100 * correct / total

if global_rank == 0:
    print(f"Test Loss: {average_test_loss:.4f}")
    print(f"Accuracy: {accuracy:.2f}%")

end = time.perf_counter()
if global_rank == 0:
    print("Training and Testing Took", end - start)

dist.destroy_process_group()
