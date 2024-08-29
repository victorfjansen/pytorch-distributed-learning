import os
import random
import time
import torch
import timm
import json
import matplotlib.pyplot as plt
import csv
import requests
from datetime import datetime

from sklearn.metrics import precision_score, recall_score


import seaborn as sns
import numpy as np

from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score

import torch.optim as optim
import torch.nn as nn
from PIL import Image

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, random_split

LOGS = []

# Função para adicionar entradas ao log
def add_log(event):
    global LOGS
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    LOGS.append([current_time, event])


def save_logs(filename):
    global LOGS
    global MODEL_NAME

    # Save the plot as a PDF
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.join(output_dir, "kaggle")
    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.join(output_dir, MODEL_NAME)
    output_dir = output_dir + '/resources'
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f'{MODEL_NAME}_SERVER_RESOURCE_LOGS.txt')
    with open(filename, 'w', newline='') as csvfile:
        log_writer = csv.writer(csvfile)
        log_writer.writerow(['Time', 'Event'])  # Cabeçalhos do CSV
        log_writer.writerows(LOGS)


def resources_usage(START, END, MACHINE_IP, IS_FIRST_MACHINE:bool):
    global MODEL_NAME
    # Convertendo as strings para objetos datetime
    train_start_time_dt = datetime.strptime(START, "%Y-%m-%d %H:%M:%S")
    train_end_time_dt = datetime.strptime(END, "%Y-%m-%d %H:%M:%S")

    # Convertendo objetos datetime para timestamps Unix
    TRAIN_START_TIME = int(time.mktime(train_start_time_dt.timetuple()))
    TRAIN_END_TIME = int(time.mktime(train_end_time_dt.timetuple()))

    # Substituindo os valores na URL
    cpu_url = f"http://{MACHINE_IP}:19999/api/v1/data?chart=system.cpu&options=unaligned&group=sum&units=percentage&after={TRAIN_START_TIME}&before={TRAIN_END_TIME}&points=3600&format=csv"

    # Coleta de RAM da ultima Hora - Disponível
    available_ram_url = f"http://{MACHINE_IP}:19999/api/v1/data?chart=mem.available&options=unaligned&group=sum&units=percentage&after={TRAIN_START_TIME}&before={TRAIN_END_TIME}&points=3600&format=csv"

    # Coleta de RAM da ultima Hora - Usada (Comprometida)
    used_ram_url = f"http://{MACHINE_IP}:19999/api/v1/data?chart=mem.committed&options=unaligned&group=sum&units=percentage&after={TRAIN_START_TIME}&before={TRAIN_END_TIME}&points=3600&format=csv"

    netpackets_url = f"http://{MACHINE_IP}:19999/api/v1/data?chart=net_packets.enp5s0f0&options=unaligned&group=avg&units=%25&after={TRAIN_START_TIME}&before={TRAIN_END_TIME}&points=86400&format=csv"

    # Coleta de GPU da ultima Hora (Comsuption)
    if IS_FIRST_MACHINE:
        gpu_url = f"http://{MACHINE_IP}:19999/api/v1/data?chart=nvidia_smi.gpu_gpu-42091774-4461-f9da-a039-2814106b5a77_gpu_utilization&options=unaligned&group=avg&units=%25&after={TRAIN_START_TIME}&before={TRAIN_END_TIME}&points=86400&format=csv"
    else:
        gpu_url = f"http://{MACHINE_IP}:19999/api/v1/data?chart=nvidia_smi.gpu_gpu-7113699b-cad5-cef4-f5fd-9ea4d34c0728_gpu_utilization&options=unaligned&group=avg&units=%25&after={TRAIN_START_TIME}&before={TRAIN_END_TIME}&points=86400&format=csv"

    # Fazendo a requisição GET
    response_cpu = requests.get(cpu_url)

    # Verificando se a requisição foi bem-sucedida
    if response_cpu.status_code == 200:
        # Salvando o conteúdo da resposta em um arquivo CSV

        # Save the plot as a PDF
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)
        output_dir = os.path.join(output_dir, "kaggle")
        os.makedirs(output_dir, exist_ok=True)
        output_dir = os.path.join(output_dir, MODEL_NAME)
        output_dir = output_dir + '/resources'
        os.makedirs(output_dir, exist_ok=True)
        cpu_file = os.path.join(output_dir, f'{MODEL_NAME}_RESOURCE_LOGS_CPU.txt')
        with open(cpu_file, 'wb') as file:
            file.write(response_cpu.content)
        print("Resource experiments (CPU Used) saved!")
    else:
        print(f"Falha na requisição. Status code: {response_cpu.status_code}")

    # Fazendo a requisição GET
    response_ram_available = requests.get(available_ram_url)

    # Verificando se a requisição foi bem-sucedida
    if response_ram_available.status_code == 200:
        # Salvando o conteúdo da resposta em um arquivo CSV

        # Save the plot as a PDF
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)
        output_dir = os.path.join(output_dir, "kaggle")
        os.makedirs(output_dir, exist_ok=True)
        output_dir = os.path.join(output_dir, MODEL_NAME)
        output_dir = output_dir + '/resources'
        os.makedirs(output_dir, exist_ok=True)
        ram_available_file = os.path.join(output_dir, f'{MODEL_NAME}_RESOURCE_LOGS_RAM_AVAILABLE.txt')

        with open(ram_available_file, 'wb') as file:
            file.write(response_ram_available.content)
        print("Resource experiments (RAM Available) saved!")
    else:
        print(f"Falha na requisição. Status code: {response_ram_available.status_code}")

        # Fazendo a requisição GET
    response_ram_used = requests.get(used_ram_url)

    # Verificando se a requisição foi bem-sucedida
    if response_ram_used.status_code == 200:
        # Salvando o conteúdo da resposta em um arquivo CSV

        # Save the plot as a PDF
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)
        output_dir = os.path.join(output_dir, "kaggle")
        os.makedirs(output_dir, exist_ok=True)
        output_dir = os.path.join(output_dir, MODEL_NAME)
        output_dir = output_dir + '/resources'
        os.makedirs(output_dir, exist_ok=True)
        ram_used_file = os.path.join(output_dir, f'{MODEL_NAME}_RESOURCE_LOGS_RAM_USED.txt')

        with open(ram_used_file, 'wb') as file:
            file.write(response_ram_used.content)
        print("Resource experiments (RAM Used) saved!")
    else:
        print(f"Falha na requisição. Status code: {response_ram_used.status_code}")

        # Fazendo a requisição GET
    response_gpu_used = requests.get(gpu_url)

    # Verificando se a requisição foi bem-sucedida
    if response_gpu_used.status_code == 200:
        # Salvando o conteúdo da resposta em um arquivo CSV

        # Save the plot as a PDF
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)
        output_dir = os.path.join(output_dir, "kaggle")
        os.makedirs(output_dir, exist_ok=True)
        output_dir = os.path.join(output_dir, MODEL_NAME)
        output_dir = output_dir + '/resources'
        os.makedirs(output_dir, exist_ok=True)
        gpu_file = os.path.join(output_dir, f'{MODEL_NAME}_RESOURCE_LOGS_GPU.txt')

        with open(gpu_file, 'wb') as file:
            file.write(response_gpu_used.content)
        print("Resource experiments (GPU Used) saved!")
    else:
        print(f"Falha na requisição. Status code: {response_gpu_used.status_code}")

    response_netpackets = requests.get(netpackets_url)

    # Verificando se a requisição foi bem-sucedida
    if response_netpackets.status_code == 200:
        # Salvando o conteúdo da resposta em um arquivo CSV

        # Save the plot as a PDF
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)
        output_dir = os.path.join(output_dir, "kaggle")
        os.makedirs(output_dir, exist_ok=True)
        output_dir = os.path.join(output_dir, MODEL_NAME)
        output_dir = output_dir + '/resources'
        os.makedirs(output_dir, exist_ok=True)
        gpu_file = os.path.join(output_dir, f'{MODEL_NAME}_RESOURCE_LOGS_NETPACKETS.txt')

        with open(gpu_file, 'wb') as file:
            file.write(response_netpackets.content)
        print("Resource experiments (NETPACKETS Used) saved!")
    else:
        print(f"Falha na requisição. Status code: {response_netpackets.status_code}")  



def format_disease_names(disease_list):
    formatted_list = []
    for disease in disease_list:
        # Substitui underscores por espaços e capitaliza cada palavra
        formatted_disease = disease.replace('_', ' ').title()
        formatted_list.append(formatted_disease)
    return formatted_list


# Custom dataset class
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}  # Dictionary to map class names to indices
        self.classes = []  # List to hold the class names

        # Populate image paths and labels
        self._load_dataset()

    def _load_dataset(self):
        # Walk through the root directory and collect image paths and labels
        for class_idx, class_dir in enumerate(sorted(os.listdir(self.root_dir))):
            class_path = os.path.join(self.root_dir, class_dir)
            if os.path.isdir(class_path):
                self.class_to_idx[class_dir] = class_idx  # Map class name to index
                self.classes.append(class_dir)  # Store class name
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

# Set environment variables for distributed training
os.environ["MASTER_ADDR"] = "200.17.78.37"  # IP address of the master node
os.environ["MASTER_PORT"] = "4500"  # Port used by the master node
os.environ["WORLD_SIZE"] = "2"  # Total number of processes across all nodes
os.environ["RANK"] = "0"  # Global rank of the current process (0 for master node)
os.environ["LOCAL_RANK"] = "0"  # Local rank on this node (0 for single GPU per machine)

# Initialize process group
dist.init_process_group("nccl")

local_rank = int(os.environ["LOCAL_RANK"])
global_rank = int(os.environ["RANK"])

is_37_machine = (global_rank == 0)

BATCH_SIZE = 32 // int(os.environ["WORLD_SIZE"])
EPOCHS = 100
WORKERS = 4
IMG_DIMS = (256, 256)
CLASSES = 13

SEED = 42

MODEL_NAME = 'mobilenetv2_100'

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

transform = transforms.Compose([
    transforms.RandomRotation(degrees=5),                      # Rotation by 5°
    transforms.RandomAffine(degrees=0, shear=0.2),             # Shear intensity of 0.2°
    transforms.RandomResizedCrop(size=IMG_DIMS, scale=(0.8, 1.0)),  # Zoom of 0.2
    transforms.RandomHorizontalFlip(),                         # Horizontal flip
    transforms.ColorJitter(),                                  # Optional: Adds color jittering
    transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.05, 0.05))]),  # Width and height shift (5%)
    transforms.ToTensor(),                                     # Convert image to Tensor
])

# Load the dataset
dataset = CustomImageDataset(root_dir="dataset/train_images", transform=transform)

# Split the dataset into training and testing subsets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DistributedSamplers for training and testing datasets
train_sampler = DistributedSampler(train_dataset)
test_sampler = DistributedSampler(test_dataset)

# Create DataLoaders for training and testing
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
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialize storage for metrics
training_metrics = {
    'loss': [],
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1_score': []
}

testing_metrics = {
    'loss': [],
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1_score': [],
    'confusion_matrix': None,
}

# Training phase
start = time.perf_counter()

print("START", start)

epochTimeStart = time.time()
add_log("EPOCH START: " + str(epochTimeStart))

TRAIN_START_TIME = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

for epoch in range(EPOCHS):
    add_log("Epoch Started: " + str(epoch))

    epoch_start_time = time.perf_counter()
    
    model.train()
    train_sampler.set_epoch(epoch)  # Ensure proper shuffling each epoch

    running_loss = 0.0
    correct = 0
    total = 0
        
    for idx, batch in enumerate(tqdm(train_loader, total=len(train_loader))):
        features, labels = batch[0].to(local_rank), batch[1].to(local_rank)

        optimizer.zero_grad()

        preds = model(features)
        loss = loss_fn(preds, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(preds, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if idx % 10 == 0:
            print("TRAIN: running_loss", running_loss/(idx+1))
            print("TRAIN: running_accuracy", correct/total)
            print("TRAIN: current epoch", epoch)
            

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct / total

    training_metrics['loss'].append(epoch_loss)
    training_metrics['accuracy'].append(epoch_accuracy)

    epoch_end_time = time.perf_counter()
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Loss: {epoch_loss:.4f} | Accuracy: {epoch_accuracy:.2f}%")
    print(f"Epoch Time: {epoch_end_time - epoch_start_time:.2f}s")
    add_log(f"Epoch Time: {epoch_end_time - epoch_start_time:.2f}s")
    add_log(f"Loss: {epoch_loss:.4f} | Accuracy: {epoch_accuracy:.2f}%")



    print("-----------------------------------------------------------------------------------")
    print("------------------------------------TEST_PHASE-------------------------------------")
    print("-----------------------------------------------------------------------------------")


    model.eval()
    test_loss = 0.0
    correct_test = 0
    total_test = 0
    
    for idx, batch in enumerate(tqdm(test_loader, total=len(test_loader))):
        features, labels = batch[0].to(local_rank), batch[1].to(local_rank)

        preds = model(features)
        loss = loss_fn(preds, labels)
        test_loss += loss.item()

        _, predicted = torch.max(preds, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()
        
        if idx % 4 == 0:
            print("TEST: running_loss", test_loss/(idx+1))
            print("TEST: running_accuracy", correct_test/total_test)

    average_test_loss = test_loss / len(test_loader)
    accuracy_test = correct_test / total_test #deveria ser total_test
    
    testing_metrics['loss'].append(average_test_loss)
    testing_metrics['accuracy'].append(accuracy_test)

TRAIN_END_TIME = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

epochTimeEnd = time.time()
add_log("EPOCH END: " + str(epochTimeEnd))

# ----------------------------------------------------------------------------------------------------------

print("-----------------------------------------------------------------------------------")
print("------------------------------------EVALUATE_THE_MODEL-------------------------------------")
print("-----------------------------------------------------------------------------------")

all_labels = []
all_predictions = []

with torch.no_grad():
    for idx, batch in enumerate(tqdm(test_loader, total=len(test_loader))):
        features, labels = batch[0].to(local_rank), batch[1].to(local_rank)

        preds = model(features)
        loss = loss_fn(preds, labels)
        test_loss += loss.item()

        _, predicted = torch.max(preds, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()
        
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

conf_matrix = confusion_matrix(all_labels, all_predictions)
testing_metrics['confusion_matrix'] = conf_matrix.tolist()  # Convert to list for JSON serialization
print(f"Confusion Matrix:\n{conf_matrix}")

# save the model
torch.save(model.state_dict(), f'test_result/{MODEL_NAME}-results.pth')

# Save metrics to JSON
with open('test_result/training_metrics.json', 'w') as f:
    json.dump(training_metrics, f)

with open('test_result/testing_metrics.json', 'w') as f:
    json.dump(testing_metrics, f)

# Plot training metrics
plt.figure(figsize=(10, 10))

# Plot loss/accuracy
plt.plot(training_metrics['loss'], label='Training Loss')
plt.plot(testing_metrics['loss'], label='Test Loss')
plt.plot(training_metrics['accuracy'], label='Training Accuracy')
plt.plot(testing_metrics['accuracy'], label='Test Accuracy')

plt.xlabel('Epochs')
plt.ylabel('Loss/Accuracy')
plt.legend(frameon=False, fontsize=12)

plt.tight_layout()
plt.savefig('test_result/training_testing_metrics.pdf')

conf_matrix = np.array(testing_metrics['confusion_matrix'])

class_names = format_disease_names(dataset.classes)

plt.figure(figsize=(10, 10))
plt.subplot(1, 1, 1)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()

plt.savefig('test_result/confusion_matrix.pdf')

end = time.perf_counter()

resources_usage(TRAIN_START_TIME, TRAIN_END_TIME, "200.17.78.37", is_37_machine)
add_log('Train Ended')
save_logs('execution_log.csv')

print("Training and Testing Took", end - start)

# Cleanup
dist.destroy_process_group()


