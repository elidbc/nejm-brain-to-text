import torch
import torch.nn as nn
import time
import os
import numpy as np
import pickle
import yaml
from torch.utils.data import DataLoader
from dataset import BrainToTextDataset, train_test_split_indicies
from exp1_model import Exp1Model
from data_augmentations import gauss_smooth
import torchaudio.functional as F

class Exp1Trainer:
    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.output_dir = self.config['experiment']['output_dir']

        os.makedirs(self.output_dir, exist_ok=True)

        # Params and Optimizer
        bias_params = [p for name, p in self.model.named_parameters() if 'bias' in name]
        adapter_params = [p for name, p in self.model.named_parameters() if 'day_adapter' in name]
        gru_params = [p for name, p in self.model.named_parameters() if 'gru_decoder' in name]
        classifier_params = [p for name, p in self.model.named_parameters() if 'classifier' in name]

        self.optimizer = torch.optim.AdamW([
            {'params': bias_params, 'weight_decay': 0.0},
            {'params': adapter_params, 'weight_decay': 0.01, 'lr': self.config['model']['lr_max_day']},
            {'params': gru_params, 'weight_decay': 0.01},
            {'params': classifier_params, 'weight_decay': 0.01},
        ])
        
        # Learning Rate Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max = self.config['experiment']['num_training_batches'],
            eta_min = self.config['model']['lr_min'],
        )

        # Loss
        self.ctc_loss = torch.nn.CTCLoss(blank = 0, reduction = 'none', zero_infinity = False)

        self.transform_args = self.config['dataset']['data_transforms']        

    def _calculate_input_lengths(self, n_time_steps):
        """
        Calculate input lengths for CTC loss
        """
        patch_size = self.config['model']['patch_size']
        patch_stride = self.config['model']['patch_stride']
        adjusted_len = ((n_time_steps - patch_size) / patch_stride + 1).floor().to(torch.int32)
        return adjusted_len

    def transform_data(self, features, n_time_steps, mode = 'train'):
        """
        Apply augmentations and smoothing to data (inherited from baseline model)
        """
        data_shape = features.shape
        batch_size = data_shape[0]
        channels = data_shape[-1]

        # We only apply these augmentations in training
        if mode == 'train':
            # add static gain noise 
            if self.transform_args['static_gain_std'] > 0:
                warp_mat = torch.tile(torch.unsqueeze(torch.eye(channels), dim = 0), (batch_size, 1, 1))
                warp_mat += torch.randn_like(warp_mat, device=self.device) * self.transform_args['static_gain_std']

                features = torch.matmul(features, warp_mat)

            # add white noise
            if self.transform_args['white_noise_std'] > 0:
                features += torch.randn(data_shape, device=self.device) * self.transform_args['white_noise_std']

            # add constant offset noise 
            if self.transform_args['constant_offset_std'] > 0:
                features += torch.randn((batch_size, 1, channels), device=self.device) * self.transform_args['constant_offset_std']

            # add random walk noise
            if self.transform_args['random_walk_std'] > 0:
                features += torch.cumsum(torch.randn(data_shape, device=self.device) * self.transform_args['random_walk_std'], dim =self.transform_args['random_walk_axis'])

            # randomly cutoff part of the data timecourse
            if self.transform_args['random_cut'] > 0:
                cut = np.random.randint(0, self.transform_args['random_cut'])
                features = features[:, cut:, :]
                n_time_steps = n_time_steps - cut

        # Apply Gaussian smoothing to data 
        # This is done in both training and validation
        if self.transform_args['smooth_data']:
            features = gauss_smooth(
                inputs = features, 
                device = self.device,
                smooth_kernel_std = self.transform_args['smooth_kernel_std'],
                smooth_kernel_size= self.transform_args['smooth_kernel_size'],
            )
            
        return features, n_time_steps

    def train(self, train_loader, val_loader):
        """
        Train the model
        """
        print("Starting training...")
        self.model.train()
        
        for batch_idx, batch in enumerate(train_loader):
            start_time = time.time()

            # 1. Move data to device
            x = batch['input_features'].to(self.device)
            labels = batch['seq_class_ids'].to(self.device)
            n_time_steps = batch['n_time_steps'].to(self.device)
            phone_seq_lens = batch['phone_seq_lens'].to(self.device)
            day_indicies = batch['day_indicies'].to(self.device)

            with torch.autocast(device_type = "cuda", enabled = self.config['experiment']['use_amp'], dtype = torch.bfloat16):
                # 2. Apply data augmentations, patching, and day-specific adapter
                x, n_time_steps = self.transform_data(x, n_time_steps, 'train')
                input_lengths = self._calculate_input_lengths(n_time_steps)

                # 3. Forward pass -> phoneme predictions
                self.optimizer.zero_grad()
                logits = self.model(x, day_indicies)

                # 4. Calculate CTC loss
                log_probs = logits.log_softmax(dim=2).permute(1, 0, 2)
                loss = self.ctc_loss(
                    log_probs = log_probs,
                    targets = labels,
                    input_lengths = input_lengths,
                    target_lengths = phone_seq_lens,
                )

                loss = torch.mean(loss) # take mean loss over batches

            # 5. Backward pass -> update weights
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
            self.optimizer.step()
            self.scheduler.step()

            # 6. Log training progress
            if batch_idx % 100 == 0:
                lr = self.scheduler.get_last_lr()[0]
                print(f"Step: {batch_idx} | Loss: {loss.item():.4f} | Lr: {lr:.6f} | Time: {time.time() - start_time:.2f}s")
            
            if batch_idx % 1000 == 0:
                val_per = self.validate(val_loader)
                self.save_checkpoint(batch_idx, val_per)
                self.model.train()

    def validate(self, val_loader):
        """
        Validate the model
        """
        self.model.eval()
        print("Running validation...")

        total_edit_distance = 0
        total_length = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                x = batch['input_features'].to(self.device)
                labels = batch['seq_class_ids'].to(self.device)
                n_time_steps = batch['n_time_steps'].to(self.device)
                phone_seq_lens = batch['phone_seq_lens'].to(self.device)
                day_indicies = batch['day_indicies'].to(self.device)

                # Data augmentation
                x, n_time_steps = self.transform_data(x, n_time_steps, 'val')
                input_lengths = self._calculate_input_lengths(n_time_steps)

                logits = self.model(x, day_indicies) # shape: (batch_size, max_seq_length, num_classes)

                preds = torch.argmax(logits, dim=2) # shape: (batch_size, max_seq_length)

                for i in range(preds.shape[0]):
                    raw_pred = preds[i, :input_lengths[i]]
                    
                    # collapse consecutive identical characters
                    pred_seq = torch.unique_consecutive(raw_pred)

                    # remove blank (0)
                    pred_seq = pred_seq[pred_seq != 0]

                    # label
                    y = labels[i, :phone_seq_lens[i]]

                    dist = F.edit_distance(pred_seq, y)
                    length = len(y)

                    total_edit_distance += dist
                    total_length += length

        per = total_edit_distance / total_length
        print(f"Validation PER: {per:.4f}")
        return per

    def save_checkpoint(self, batch_idx, val_per):
        """
        Save the model checkpoint
        """
        path = os.path.join(self.output_dir, f"checkpoint_batch_{batch_idx}.pt")
        torch.save({
            'batch_idx': batch_idx,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'val_per': val_per,
        }, path)
        print(f"Saved checkpoint to {path}")


def load_config(yaml_path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # 1. Load Configuration
    config = load_config('exp1_args.yaml')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Prepare Data Splits
    data_dir = config['dataset']['dataset_dir']
    sessions = config['dataset']['sessions']
    file_paths = [os.path.join(data_dir, s, 'data_train.hdf5') for s in sessions]
    
    # Create train/val splits using dataset.py
    train_trials, val_trials = train_test_split_indicies(
        file_paths=file_paths,
        test_percentage=config['dataset']['test_percentage'],
        seed=config['dataset']['seed']
    )

    # 3. Initialize Datasets
    # CRITICAL: days_per_batch is handled inside the Dataset class
    train_ds = BrainToTextDataset(
        trial_indicies=train_trials,
        split='train',
        days_per_batch=config['dataset']['days_per_batch'],
        n_batches=config['experiment']['num_training_batches'],
        batch_size=config['dataset']['batch_size'],
        must_include_days=None,
        random_seed=config['experiment']['seed'],
        feature_subset=None,
    )

    val_ds = BrainToTextDataset(
        trial_indicies=val_trials,
        split='test',
        days_per_batch=None,
        n_batches=None,
        batch_size=config['dataset']['batch_size'],
        must_include_days=None,
        random_seed=config['experiment']['seed'],
        feature_subset=None,
    )

    # 4. Initialize DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=None, 
        shuffle=config['dataset']['loader_shuffle'],
        num_workers=config['dataset']['num_dataloader_workers'],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=None,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    print(f"Initialized datasets and data loaders")

    # 5. Initialize Model, Trainer, and Start Training
    model = Exp1Model(config, num_days=len(sessions))
    trainer = Exp1Trainer(model, config, device)
    trainer.train(train_loader, val_loader)

if __name__ == "__main__":
    main()

