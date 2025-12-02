import torch
from torch import nn

class DayAdapter(nn.Module):
    """
    Day-specific adapter MLP to project neural features from different days
    to a common latent space for the GRU.
    """
    def __init__(self, input_dim, hidden_dim, dropout_rate, output_dim, num_days):
        super().__init__()
        self.num_days = num_days

        # One adapter per day
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            ) for _ in range(num_days)
        ])

    def forward(self, x, day_indices):
        """
        x: (batch_size, time, input_dim)
        day_indices: (batch_size,)
        """
        out = torch.zeros_like(x)
        unique_days = torch.unique(day_indices)

        for day_idx in unique_days:
            mask = day_indices == day_idx
            adapter_out = self.adapters[day_idx](x[mask])
            out[mask] = adapter_out.to(dtype=x.dtype)

        return out


class GRUDecoder(nn.Module):
    """
    GRU decoder to process sequence data.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate, bidirectional):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=bidirectional
        )

    def forward(self, x):
        out, _ = self.gru(x)
        return out


class Exp2Model(nn.Module):
    """
    Primary model combining DayAdapter and pretrained GRU.
    Handles weight loading and optional freezing of GRU.
    """
    def __init__(self, config, num_days, pretrained_ckpt_path=None, freeze_gru=True):
        super().__init__()
        self.config = config

        # --- Day Adapter ---
        self.day_adapter = DayAdapter(
            input_dim=config['model']['adapter']['neural_dim'],
            hidden_dim=config['model']['adapter']['hidden_dim'],
            dropout_rate=config['model']['adapter']['dropout_rate'],
            output_dim=config['model']['adapter']['output_dim'],
            num_days=num_days
        )

        # --- Patch size and stride ---
        self.patch_size = config['model']['patch_size']
        self.patch_stride = config['model']['patch_stride']
        patched_dim = config['model']['adapter']['output_dim'] * self.patch_size
        gru_input_dim = patched_dim

        # --- GRU Decoder ---
        self.gru_decoder = GRUDecoder(
            input_dim=gru_input_dim,
            hidden_dim=config['model']['gru_decoder']['hidden_dim'],
            num_layers=config['model']['gru_decoder']['num_layers'],
            dropout_rate=config['model']['gru_decoder']['dropout_rate'],
            bidirectional=config['model']['gru_decoder']['bidirectional'],
        )

        gru_out_dim = config['model']['gru_decoder']['hidden_dim'] * \
            (2 if config['model']['gru_decoder']['bidirectional'] else 1)

        # --- Classifier ---
        self.classifier = nn.Linear(gru_out_dim, config['model']['num_classes'])

        # --- Load pretrained weights ---
        if pretrained_ckpt_path is not None:
            self.load_pretrained_weights(pretrained_ckpt_path)

        # --- Optionally freeze GRU and classifier ---
        if freeze_gru:
            for name, param in self.named_parameters():
                if 'gru_decoder' in name or 'classifier' in name:
                    param.requires_grad = False

        # --- Optional: initialize DayAdapter near identity ---
        for adapter in self.day_adapter.adapters:
            nn.init.eye_(adapter[0].weight)
            nn.init.zeros_(adapter[0].bias)

    def load_pretrained_weights(self, ckpt_path):
        """
        Load pretrained baseline model weights for GRU + classifier only.
        
        The baseline model (rnn_model.py) has a different structure:
        - Baseline: _orig_mod.gru.* → Exp2: gru_decoder.gru.*
        - Baseline: _orig_mod.out.* → Exp2: classifier.*
        - Baseline: _orig_mod.h0 → stored for learnable initial hidden state (not used in Exp2)
        - Baseline: _orig_mod.day_weights/day_biases → skipped (Exp2 uses different adapter)
        """
        baseline_ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        baseline_state_dict = baseline_ckpt['model_state_dict']
        current_state_dict = self.state_dict()
        
        # Build mapping from baseline keys to exp2 keys
        loaded_keys = []
        for baseline_key, baseline_val in baseline_state_dict.items():
            # Strip _orig_mod. prefix if present (from torch.compile)
            clean_key = baseline_key.replace('_orig_mod.', '')
            
            # Skip day-specific weights (using different adapter architecture)
            if clean_key.startswith('day_weights') or clean_key.startswith('day_biases'):
                continue
                
            # Skip h0 (learnable initial hidden state) - not used in Exp2
            if clean_key == 'h0':
                continue
            
            # Map GRU weights: gru.* → gru_decoder.gru.*
            if clean_key.startswith('gru.'):
                exp2_key = 'gru_decoder.' + clean_key
            # Map classifier: out.* → classifier.*
            elif clean_key.startswith('out.'):
                exp2_key = clean_key.replace('out.', 'classifier.')
            else:
                continue  # Skip any other keys
            
            # Validate shape matches
            if exp2_key in current_state_dict:
                if current_state_dict[exp2_key].shape == baseline_val.shape:
                    current_state_dict[exp2_key] = baseline_val
                    loaded_keys.append(f"{baseline_key} → {exp2_key}")
                else:
                    print(f"  Shape mismatch for {exp2_key}: "
                          f"expected {current_state_dict[exp2_key].shape}, "
                          f"got {baseline_val.shape}")
            else:
                print(f"  Key {exp2_key} not found in Exp2Model")
        
        self.load_state_dict(current_state_dict)
        print(f"Loaded {len(loaded_keys)} pretrained weights from {ckpt_path}")
        for k in loaded_keys:
            print(f"  {k}")

    def _apply_patching(self, x):
        """
        Apply sliding-window patching along time dimension
        """
        if self.patch_size <= 0:
            return x

        x = x.unsqueeze(1)                   # [B,1,T,F]
        x = x.permute(0, 3, 1, 2)            # [B,F,1,T]
        x_unfold = x.unfold(3, self.patch_size, self.patch_stride)
        x_unfold = x_unfold.squeeze(2)       # remove dummy dim
        x_unfold = x_unfold.permute(0, 2, 3, 1)
        x = x_unfold.reshape(x.size(0), x_unfold.size(1), -1)
        return x

    def forward(self, x, day_idx):
        """
        Forward pass:
        1. Apply day adapter
        2. Apply patching
        3. GRU decoder
        4. Classifier
        """
        x = self.day_adapter(x, day_idx)
        x = self._apply_patching(x)
        x = self.gru_decoder(x)
        logits = self.classifier(x)
        return logits