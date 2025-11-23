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
from evaluate_model_helpers import LOGIT_TO_PHONEME
import torchaudio.functional as F



class Exp1Trainer:
    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.output_dir = self.config['experiment']['output_dir']
        self.transform_args = self.config['dataset']['data_transforms']

        os.makedirs(self.output_dir, exist_ok=True)

        # data
        self.data_dir = config['dataset']['dataset_dir']
        self.sessions = config['dataset']['sessions'][:1]
        self.file_paths = [os.path.join(self.data_dir, s, 'data_train.hdf5') for s in self.sessions]

        """train_trials, _ = train_test_split_indicies(
            file_paths=self.file_paths,
            test_percentage=0,
            seed=config['dataset']['seed']
        )

        _, val_trials = train_test_split_indicies(
            file_paths=self.file_paths,
            test_percentage=1,
            seed=config['dataset']['seed']
        )"""

        train_trials, val_trials = train_test_split_indicies(
            file_paths=self.file_paths,
            test_percentage=config['dataset']['test_percentage'],
            seed=config['dataset']['seed']
        )

        self.train_ds = BrainToTextDataset(
            trial_indicies=train_trials,
            split='train',
            days_per_batch = config['dataset']['days_per_batch'],
            n_batches = config['experiment']['num_training_batches'],
            batch_size = config['dataset']['batch_size'],
            must_include_days = config['dataset']['must_include_days'],
            random_seed = config['experiment']['seed'],
            feature_subset = config['dataset']['feature_subset'],
        )
        self.train_loader = DataLoader(
            self.train_ds,
            batch_size = None,
            shuffle = self.config['dataset']['loader_shuffle'],
            num_workers = self.config['dataset']['num_dataloader_workers'],
            pin_memory = True
        )

        # val dataset and dataloader
        self.val_dataset = BrainToTextDataset(
            trial_indicies = val_trials, 
            split = 'test',
            days_per_batch = config['dataset']['days_per_batch'],
            n_batches = None,
            batch_size = self.config['dataset']['batch_size'],
            must_include_days = None,
            random_seed = self.config['dataset']['seed'],
            feature_subset = self.config['dataset']['feature_subset']   
            )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size = None,
            shuffle = False, 
            num_workers = 0,
            pin_memory = True 
        )

        # optimizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config['model']['lr_min'])
        #self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.0001)

        # objective
        self.ctc_loss = torch.nn.CTCLoss(blank = 0, reduction = 'none', zero_infinity = False)


########################################################
# Data Helper Functions
########################################################
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

########################################################
# Training Loop
########################################################

    def train(self):
        """
        Train the model
        """
        self.model.train()
        
        for i, batch in enumerate(self.train_loader):
            

            x = batch['input_features'].to(self.device)
            labels = batch['seq_class_ids'].to(self.device)
            n_time_steps = batch['n_time_steps'].to(self.device)
            phone_seq_lens = batch['phone_seq_lens'].to(self.device)
            day_indicies = batch['day_indicies'].to(self.device)

            x, n_time_steps = self.transform_data(x, n_time_steps, 'train')
            adjusted_len = self._calculate_input_lengths(n_time_steps)

            logits = self.model(x, day_indicies)
            log_probs = logits.log_softmax(dim=2).permute(1, 0, 2)

            loss = self.ctc_loss(
                log_probs = log_probs,
                targets = labels,
                input_lengths = adjusted_len,
                target_lengths = phone_seq_lens
            )
            loss = torch.mean(loss)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
            self.optimizer.step()
            #self.scheduler.step()

            if i % 10 == 0:
                #current_lr = self.scheduler.get_last_lr()[0]
                print(f"Batch {i} | Loss: {loss.item():.4f}")

            if i > 1190:
                print(f"Batch {i} | Loss: {loss.item():.4f}")
                
                # 1. Get the most likely phoneme for each time step (Argmax)
                pred_indices = torch.argmax(logits[0], dim=1).cpu().numpy() # Take first item in batch
                target_indices = labels[0, :phone_seq_lens[0]].cpu().numpy()
                
                # 2. Convert to strings (simple decode, no collapse)
                # We filter out 0 (Blank) to see if it's predicting ANYTHING other than silence
                pred_str = " ".join([LOGIT_TO_PHONEME[idx] for idx in pred_indices if idx != 0])
                target_str = " ".join([LOGIT_TO_PHONEME[idx] for idx in target_indices])
                
                print(f"Target: {target_str}")
                print(f"Pred  : {pred_str}")
                print("------------------------------------------------")

            if i > 0 and i % 50 == 0:
                self.val_debug()
                self.model.train() # Switch back to train mode!
        
    def val_debug(self):
        self.model.eval()
        val_loss_accum = 0
        num_batches = 0
        print("------- Validating -------")

        with torch.no_grad():
            for batch in self.val_loader:
                x = batch['input_features'].to(self.device)
                labels = batch['seq_class_ids'].to(self.device)
                n_time_steps = batch['n_time_steps'].to(self.device)
                phone_seq_lens = batch['phone_seq_lens'].to(self.device)
                day_indicies = batch['day_indicies'].to(self.device)

                x, n_time_steps = self.transform_data(x, n_time_steps, 'val')
                adjusted_len = self._calculate_input_lengths(n_time_steps)

                logits = self.model(x, day_indicies)
                log_probs = logits.log_softmax(dim=2).permute(1, 0, 2)
                
                loss = self.ctc_loss(
                    log_probs = log_probs,
                    targets = labels,
                    input_lengths = adjusted_len,
                    target_lengths = phone_seq_lens
                )
                val_loss_accum += torch.mean(loss).item()
                num_batches += 1
        avg_val_loss = val_loss_accum / max(1, num_batches)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print("--------------------------")
        



    def validate(self, val_loader):
        """
        Validate the model
        """
        self.model.eval()
        print(f"--------------------------------")
        print("Running validation...")

        total_edit_distance = 0
        total_length = 0
        total_val_loss = 0
        num_batches = 0

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

                log_probs = logits.log_softmax(dim=2).permute(1, 0, 2)
                loss = self.ctc_loss(
                    log_probs = log_probs,
                    targets = labels,
                    input_lengths = input_lengths,
                    target_lengths = phone_seq_lens,
                )
                total_val_loss += torch.mean(loss).item() # take mean loss over batches
                num_batches += 1

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

        avg_per = total_edit_distance / total_length
        avg_val_loss = total_val_loss / num_batches
        print(f"Validation PER: {avg_per:.4f} | Validation Loss: {avg_val_loss:.4f}")
        return avg_per, avg_val_loss

    def save_checkpoint(self, batch_idx, val_per, filename):
        """
        Save the model checkpoint
        """
        path = os.path.join(self.output_dir, filename)
        torch.save({
            'batch_idx': batch_idx,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'val_per': val_per,
            'best_val_per': self.best_val_per,
        }, path)
        #print(f"Saved checkpoint to {path}")


def load_config(yaml_path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def debug_trainer():
    config = load_config('debug.yaml')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = Exp1Model(config, num_days=1)
    print(f"Model initialized | Number of parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Number of adapters: {len(model.day_adapter.adapters)} | Adapter parameters: {sum(p.numel() for p in model.day_adapter.adapters.parameters())}")
    print(f"Gru parameters: {sum(p.numel() for p in model.gru_decoder.parameters())}")
    print(f"Classifier parameters: {sum(p.numel() for p in model.classifier.parameters())}")

    trainer = Exp1Trainer(model, config, device)
    trainer.train()
    return
    




def main():
    # 1. Load Configuration
    debug_trainer()
    """config = load_config('exp1_args.yaml')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Prepare Data Splits
    data_dir = config['dataset']['dataset_dir']
    # try with only one session
    sessions = config['dataset']['sessions'][:1]
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
        days_per_batch=min(len(sessions), config['dataset']['days_per_batch']),
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
    #model = Exp1Model(config, num_days=config['dataset']['n_sessions'])
    model = Exp1Model(config, num_days=1)
    print(f"Initialized model — number of parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Adapter parameters: {sum(p.numel() for p in model.day_adapter.adapters.parameters())}")
    print(f"Gru parameters: {sum(p.numel() for p in model.gru_decoder.parameters())}")
    print(f"Classifier parameters: {sum(p.numel() for p in model.classifier.parameters())}")
    trainer = Exp1Trainer(model, config, device)
    trainer.train(train_loader, val_loader)"""

if __name__ == "__main__":
    main()

