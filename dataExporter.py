import os
import time
import json
import csv
import requests

def resources_usage( MACHINE_IP, TRAIN_START_TIME, TRAIN_END_TIME, MODEL_NAME, IS_FIRST_MACHINE):
    # Substituindo os valores na URL
    cpu_url = f"http://{MACHINE_IP}:19999/api/v1/data?chart=system.cpu&options=unaligned&units=percentage&after={TRAIN_START_TIME}&before={TRAIN_END_TIME}&points=3600&format=csv"

    # Coleta de RAM da ultima Hora - Disponível
    available_ram_url = f"http://{MACHINE_IP}:19999/api/v1/data?chart=mem.available&options=unaligned&units=percentage&after={TRAIN_START_TIME}&before={TRAIN_END_TIME}&points=3600&format=csv"

    # Coleta de RAM da ultima Hora - Usada (Comprometida)
    used_ram_url = f"http://{MACHINE_IP}:19999/api/v1/data?chart=mem.committed&options=unaligned&units=percentage&after={TRAIN_START_TIME}&before={TRAIN_END_TIME}&points=3600&format=csv"

    # Coleta de GPU da ultima Hora (Comsuption)
    if IS_FIRST_MACHINE:
        gpu_url = f"http://{MACHINE_IP}:19999/api/v1/data?chart=nvidia_smi.gpu_gpu-42091774-4461-f9da-a039-2814106b5a77_gpu_utilization&options=unaligned&units=%25&after={TRAIN_START_TIME}&before={TRAIN_END_TIME}&points=86400&format=csv"
    else:
        gpu_url = f"http://{MACHINE_IP}:19999/api/v1/data?chart=nvidia_smi.gpu_gpu-7113699b-cad5-cef4-f5fd-9ea4d34c0728_gpu_utilization&options=unaligned&units=%25&after={TRAIN_START_TIME}&before={TRAIN_END_TIME}&points=86400&format=csv"

    netpackets_url = f"http://{MACHINE_IP}:19999/api/v1/data?chart=net_packets.enp5s0f0&options=unaligned&units=%25&after={TRAIN_START_TIME}&before={TRAIN_END_TIME}&points=86400&format=csv"

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
        cpu_file = os.path.join(output_dir, f'{MODEL_NAME}_RESOURCE_LOGS_CPU.csv')
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
        ram_available_file = os.path.join(output_dir, f'{MODEL_NAME}_RESOURCE_LOGS_RAM_AVAILABLE.csv')

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
        ram_used_file = os.path.join(output_dir, f'{MODEL_NAME}_RESOURCE_LOGS_RAM_USED.csv')

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
        gpu_file = os.path.join(output_dir, f'{MODEL_NAME}_RESOURCE_LOGS_GPU.csv')

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
        gpu_file = os.path.join(output_dir, f'{MODEL_NAME}_RESOURCE_LOGS_NETPACKETS.csv')

        with open(gpu_file, 'wb') as file:
            file.write(response_gpu_used.content)
        print("Resource experiments (NETPACKETS Used) saved!")
    else:
        print(f"Falha na requisição. Status code: {response_netpackets.status_code}")    
    
resources_usage('200.17.78.37', '1724859724.5936084', '1724898230.1748636', 'mobilenetv2_100-augmentation-200-17-78-37', True)
resources_usage('200.17.78.38', '1724859724.5876665', '1724898231.1857677', 'mobilenetv2_100-augmentation-200-17-78-38', False)

    
