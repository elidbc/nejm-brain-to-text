import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import BrainToTextDataset

class Exp1Trainer:
    def __init__(self, model, config, device, args):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.args = args
   
        if torch.cuda.is_available():
            gpu_num = self.args.get('gpu_number', 0)
            try:
                gpu_num = int(gpu_num)
            except ValueError:
                self.logger.warning(f"Invalid gpu_number value: {gpu_num}. Using 0 instead.")
                gpu_num = 0

            max_gpu_index = torch.cuda.device_count() - 1
            if gpu_num > max_gpu_index:
                self.logger.warning(f"Requested GPU {gpu_num} not available. Using GPU 0 instead.")
                gpu_num = 0

            try:
                self.device = torch.device(f"cuda:{gpu_num}")
                test_tensor = torch.tensor([1.0]).to(self.device)
                test_tensor = test_tensor * 2
            except Exception as e:
                self.logger.error(f"Error initializing CUDA device {gpu_num}: {str(e)}")
                self.logger.info("Falling back to CPU")
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        self.logger.info(f'Using device: {self.device}')

        self.model = Exp1Model(self.config, self.args['dataset']['n_sessions'])

    def train(self, train_loader, val_loader):
        pass

    def validate(self, val_loader):
        pass