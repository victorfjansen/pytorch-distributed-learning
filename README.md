# Paddy Disease Classification

Este repositório contém o código para treinar um modelo de classificação de doenças em imagens de arroz utilizando PyTorch e Distributed Data Parallel (DDP).

## Pré-requisitos

- Python 3.x
- Ambiente configurado com `torch`, `torchvision`, `torchaudio`, e `timm` (instalação automática via `requirements.txt`)

## Configuração do Dataset

1. Baixe o dataset de imagens de arroz.
2. Extraia o conteúdo do arquivo zip na pasta `dataset/`.

## Treinamento
Para iniciar o treinamento usando PyTorch Distributed Data Parallel (DDP), execute o seguinte comando:

```python3 train_ddp_wvc.py```

## Configuração de Treinamento Distribuído
O script train_ddp_wvc.py identifica automaticamente o IP da máquina principal (nó master) e define as variáveis de ambiente necessárias para o treinamento distribuído.

Na máquina principal, as variáveis serão configuradas assim dentro do arquivo:


```python
os.environ["MASTER_ADDR"] = "IP_DA_MAQUINA_PRINCIPAL"
os.environ["MASTER_PORT"] = "4500"
os.environ["WORLD_SIZE"] = "NÚMERO_TOTAL_DE_MAQUINAS"
os.environ["RANK"] = "0"  # A máquina principal sempre terá rank 0
os.environ["LOCAL_RANK"] = "0"
```

Nas demais máquinas, o script será modificado para que elas tenham o mesmo *MASTER_ADDR*, mas com *RANK* diferente entre elas. Não iremos mexer no LOCAL_RANK ainda.

Exemplo para a segunda máquina (RANK 1):

```python
os.environ["MASTER_ADDR"] = "IP_DA_MAQUINA_PRINCIPAL"
os.environ["MASTER_PORT"] = "4500"
os.environ["WORLD_SIZE"] = "NÚMERO_TOTAL_DE_MAQUINAS"
os.environ["RANK"] = "1"  # RANK correspondente para a segunda máquina
os.environ["LOCAL_RANK"] = "0"
```
