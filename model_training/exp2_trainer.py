import torch
import torch.nn as nn
import time
import os
import numpy as np
import pickle
import yaml
import wandb
from torch.utils.data import DataLoader
from dataset import BrainToTextDataset, train_test_split_indicies
from exp2_model import Exp2Model
from data_augmentations import gauss_smooth
from evaluate_model_helpers import LOGIT_TO_PHONEME
import torchaudio.functional as F

class Exp2Trainer:
    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.output_dir = self.config['experiment']['output_dir']
        self.freeze_gru = self.config['experiment'].get('freeze_gru', False)
        
        self.wandb_run = wandb.init(
            project="brain-to-text-exp2",
            config=config,
            name=self.config['experiment']['name']
        )

        os.makedirs(self.output_dir, exist_ok=True)

        # Gradient Accumulation Settings
        # Compute at init time so scheduler T_max is set correctly
        # Effective batch size = target_batch_size, achieved by accumulating gradients
        # over (target_batch_size / cuda_batch_size) mini-batches
        target_batch_size = self.config['experiment']['target_batch_size']
        cuda_batch_size = self.config['experiment']['cuda_batch_size']
        self.accumulation_steps = target_batch_size // cuda_batch_size
        # Number of optimizer steps = num_training_batches / accumulation_steps
        self.num_optimizer_steps = self.config['experiment']['num_training_batches'] // self.accumulation_steps
        print(f"Gradient accumulation: {self.accumulation_steps} steps, {self.num_optimizer_steps} optimizer updates")

        # Params and Optimizer - only include trainable parameters
        # When freeze_gru=True, GRU and classifier params have requires_grad=False
        trainable_params = [(name, p) for name, p in self.model.named_parameters() if p.requires_grad]
        
        # Build parameter groups based on freeze_gru setting
        lr_adapter = self.config['model']['lr_max_day']
        lr_gru = self.config['model'].get('lr_max_gru', lr_adapter * 0.1)
        
        # Separate params by component and bias/weight for proper weight decay
        adapter_bias_params = [p for name, p in trainable_params if 'day_adapter' in name and 'bias' in name]
        adapter_weight_params = [p for name, p in trainable_params if 'day_adapter' in name and 'bias' not in name]
        gru_bias_params = [p for name, p in trainable_params if 'gru_decoder' in name and 'bias' in name]
        gru_weight_params = [p for name, p in trainable_params if 'gru_decoder' in name and 'bias' not in name]
        classifier_bias_params = [p for name, p in trainable_params if 'classifier' in name and 'bias' in name]
        classifier_weight_params = [p for name, p in trainable_params if 'classifier' in name and 'bias' not in name]
        
        # Build param groups list
        param_groups = [
            {'params': adapter_bias_params, 'weight_decay': 0.0, 'lr': lr_adapter},
            {'params': adapter_weight_params, 'weight_decay': 0.01, 'lr': lr_adapter},
        ]
        
        if self.freeze_gru:
            print("GRU frozen: training only day_adapter parameters")
            print(f"  Adapter LR: {lr_adapter}")
        else:
            print("Training all parameters (GRU not frozen)")
            print(f"  Adapter LR: {lr_adapter}, GRU/Classifier LR: {lr_gru}")
            param_groups.extend([
                {'params': gru_bias_params, 'weight_decay': 0.0, 'lr': lr_gru},
                {'params': gru_weight_params, 'weight_decay': 0.01, 'lr': lr_gru},
                {'params': classifier_bias_params, 'weight_decay': 0.0, 'lr': lr_gru},
                {'params': classifier_weight_params, 'weight_decay': 0.01, 'lr': lr_gru},
            ])
        
        # Filter out empty param groups
        param_groups = [pg for pg in param_groups if len(pg['params']) > 0]
        
        # Create optimizer
        self.optimizer = torch.optim.AdamW(param_groups)
        
        # Log parameter counts
        print(f"  Adapter params: {sum(p.numel() for p in adapter_bias_params + adapter_weight_params):,}")
        if not self.freeze_gru:
            print(f"  GRU params: {sum(p.numel() for p in gru_bias_params + gru_weight_params):,}")
            print(f"  Classifier params: {sum(p.numel() for p in classifier_bias_params + classifier_weight_params):,}")
        
        # Learning Rate Scheduler
        # T_max is set to num_optimizer_steps (not num_training_batches) because
        # scheduler.step() is called once per gradient accumulation cycle, not once per batch
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max = self.num_optimizer_steps,
            eta_min = self.config['model']['lr_min'],
        )

        # Loss
        self.ctc_loss = torch.nn.CTCLoss(blank = 0, reduction = 'none', zero_infinity = False)
        self.transform_args = self.config['dataset']['data_transforms']

        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_per': [],
            'steps': [],
        }
        self.best_val_per = float('inf')

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

        # Use accumulation_steps computed at init time (ensures scheduler T_max is consistent)
        accumulation_steps = self.accumulation_steps
        print(f"Using gradient accumulation with {accumulation_steps} steps")
        print(f"Target batch size: {self.config['experiment']['target_batch_size']} | CUDA batch size: {self.config['experiment']['cuda_batch_size']}")
        
        self.optimizer.zero_grad()
        
        # Initialize loss accumulator
        total_loss = 0
        num_batches = 0
        step_counter = 0

        for batch_idx, batch in enumerate(train_loader):
            start_time = time.time()

            # 1. Move data to device
            x = batch['input_features'].to(self.device)
            labels = batch['seq_class_ids'].to(self.device)
            n_time_steps = batch['n_time_steps'].to(self.device)
            phone_seq_lens = batch['phone_seq_lens'].to(self.device)
            day_indicies = batch['day_indicies'].to(self.device)

            with torch.autocast(device_type = "cuda", enabled = self.config['experiment']['use_amp'], dtype = torch.float16):
                # 2. Apply data augmentations, patching, and day-specific adapter
                x, n_time_steps = self.transform_data(x, n_time_steps, 'train')
                input_lengths = self._calculate_input_lengths(n_time_steps)

                # 3. Forward pass -> phoneme predictions
                #self.optimizer.zero_grad()
                logits = self.model(x, day_indicies)

                # 4. Calculate CTC loss
                log_probs = logits.log_softmax(dim=2).permute(1, 0, 2)
                loss = self.ctc_loss(
                    log_probs = log_probs,
                    targets = labels,
                    input_lengths = input_lengths,
                    target_lengths = phone_seq_lens,
                )

                loss = torch.mean(loss) / accumulation_steps # take mean loss over batches


            # 5. Backward pass -> update weights
            loss.backward()
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                step_counter += 1

            # Log training progress every 100 batches
            if (batch_idx + 1) % 100 == 0:
                lrs = self.scheduler.get_last_lr()
                lr_adapter = lrs[0]
                lr_gru = lrs[2] if not self.freeze_gru else 0.0
                raw_loss = loss.item() * accumulation_steps
                self.history['train_loss'].append(raw_loss)
                wandb.log({
                    "train_loss": raw_loss,
                    "lr_adapter": lr_adapter,
                    "lr_gru": lr_gru,
                    "batch": batch_idx + 1,
                    "step": step_counter,
                })
                print(f"Batch {batch_idx + 1:>5} | Loss: {raw_loss:.4f} | Step {step_counter:>4} | AdpLR: {lr_adapter:.6f} | GruLR: {lr_gru:.6f}")

            # Run validation every 300 batches
            if (batch_idx + 1) % 300 == 0:
                val_per, val_loss = self.validate(val_loader)

                self.history['val_loss'].append(val_loss)
                self.history['val_per'].append(val_per)
                self.history['steps'].append(batch_idx)
                
                self.save_checkpoint(batch_idx, val_per, filename="checkpoint_latest.pt")

                if val_per < self.best_val_per:
                    print(f"New best PER: {val_per:.4f}")
                    self.best_val_per = val_per
                    self.save_checkpoint(batch_idx, val_per, filename="checkpoint_best.pt")

                with open(os.path.join(self.output_dir, 'history.pkl'), 'wb') as f:
                    pickle.dump(self.history, f)
                self.model.train()

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
        printed_sample = False  # Flag to print one sample per validation

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
                    
                    # Print one sample prediction per validation to monitor CTC behavior
                    if not printed_sample:
                        # Convert IDs to phoneme names
                        raw_pred_phonemes = [LOGIT_TO_PHONEME[p.item()] for p in raw_pred[:50]]  # First 50 raw preds
                        pred_phonemes = [LOGIT_TO_PHONEME[p.item()] for p in pred_seq]
                        label_phonemes = [LOGIT_TO_PHONEME[p.item()] for p in y]
                        
                        # Count blanks in raw prediction
                        blank_count = (raw_pred == 0).sum().item()
                        blank_pct = 100 * blank_count / len(raw_pred)
                        
                        print(f"Sample prediction (first 50 raw): {' '.join(raw_pred_phonemes)}")
                        print(f"  Blanks: {blank_count}/{len(raw_pred)} ({blank_pct:.1f}%)")
                        print(f"  Decoded prediction: {' '.join(pred_phonemes)}")
                        print(f"  Ground truth:       {' '.join(label_phonemes)}")
                        printed_sample = True

        avg_per = total_edit_distance / total_length
        avg_val_loss = total_val_loss / num_batches
        wandb.log({
            "val_loss": avg_val_loss,
            "val_per": avg_per
        })
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

def main():
    # 1. Load Configuration
    config = load_config('exp2_args.yaml')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Prepare Data Splits
    # CHANGED: Use separate data_train.hdf5 and data_val.hdf5 files to match baseline trainer
    # This ensures validation is on truly held-out data (same split as baseline experiments)
    # Previously: split data_train.hdf5 into train/val using test_percentage
    data_dir = config['dataset']['dataset_dir']
    sessions = config['dataset']['sessions']
    
    # Training data: use all trials from data_train.hdf5 (test_percentage=0 means keep all for train)
    train_file_paths = [os.path.join(data_dir, s, 'data_train.hdf5') for s in sessions]
    train_trials, _ = train_test_split_indicies(
        file_paths=train_file_paths,
        test_percentage=0,  # Keep all trials for training
        seed=config['dataset']['seed']
    )
    
    # Validation data: use all trials from data_val.hdf5 (test_percentage=1 means all go to val)
    val_file_paths = [os.path.join(data_dir, s, 'data_val.hdf5') for s in sessions]
    _, val_trials = train_test_split_indicies(
        file_paths=val_file_paths,
        test_percentage=1,  # All trials go to validation
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

    # 5. Initialize Model with optional pretrained weights
    ckpt_type = config['experiment'].get('ckpt_type', 'pretrained')
    ckpt_path = config['experiment'].get('pretrained_ckpt_path', None) if ckpt_type == 'pretrained' else config['experiment'].get('mlp_ckpt_path', None)
    freeze_gru = config['experiment'].get('freeze_gru', False)
    
    model = Exp2Model(
        config=config, 
        num_days=config['dataset']['n_sessions'],
        pretrained_ckpt_path=ckpt_path,
        ckpt_type=ckpt_type,
        freeze_gru=freeze_gru
    )
    
    # Report parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"Initialized Exp2Model:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters: {frozen_params:,}")
    print(f"  Adapter parameters: {sum(p.numel() for p in model.day_adapter.adapters.parameters()):,}")
    print(f"  GRU parameters: {sum(p.numel() for p in model.gru_decoder.parameters()):,}")
    print(f"  Classifier parameters: {sum(p.numel() for p in model.classifier.parameters()):,}")
    
    # 6. Initialize Trainer and Start Training
    trainer = Exp2Trainer(model, config, device)
    trainer.train(train_loader, val_loader)

if __name__ == "__main__":
    main()

