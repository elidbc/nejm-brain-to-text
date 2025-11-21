import torch
from torch import nn

class DayAdapter(nn.Module):
    """
    Interface for data -> GRU unit
    Improves over baseline model with a non-linearity
    in addition to a learnable day-specific weight matrix
    """
    def __init__(self, input_dim, hidden_dim, dropout_rate, output_dim, num_days):
        super().__init__()
        self.num_days = num_days

        self.adapters = nn.ModuleList([nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        ) for _ in range(num_days)
        ])

    def forward(self, x, day_indicies):
        """
        x: (batch_size, time, input_dim)
        day_indicies: (batch_size,)
        """
        out = torch.zeros_like(x)
        unique_days = torch.unique(day_indicies)

        for day_idx in unique_days:
            mask = day_indicies == day_idx
            out[mask] = self.adapters[day_idx](x[mask])

        return out

class GRUDecoder(nn.Module):
    """
    GRU Model. Process Sentence level temporal data
    """
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate, bidirectional):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=bidirectional,
        )
    
    def forward(self, x):
        """
        x: (batch_size, time, input_dim)
        """
        out, _ = self.gru(x)
        return out

class Exp1Model(nn.Module):
    """
    Primary Model. Combines DayAdapter and GRUDecoder
    """
    def __init__(self, config, num_days):
        super().__init__()
        self.config = config

        # Day-specific Adapter
        self.day_adapter = DayAdapter(
            input_dim=self.config['model']['adapter']['neural_dim'],
            hidden_dim=self.config['model']['adapter']['hidden_dim'],
            dropout_rate=self.config['model']['adapter']['dropout_rate'],
            output_dim=self.config['model']['adapter']['output_dim'],
            num_days=num_days,
        )

        # Input Processing (patching)
        self.patch_size = self.config['model']['patch_size']
        self.patch_stride = self.config['model']['patch_stride']
        patched_dim = self.config['model']['neural_dim'] * self.patch_size
        gru_input_dim = patched_dim

        # GRU Decoder
        self.gru_decoder = GRUDecoder(
            input_dim=gru_input_dim,
            hidden_dim=self.config['model']['gru_decoder']['hidden_dim'],
            num_layers=self.config['model']['gru_decoder']['num_layers'],
            dropout_rate=self.config['model']['gru_decoder']['dropout_rate'],
            bidirectional=self.config['model']['gru_decoder']['bidirectional'],
        )

        gru_out_dim = self.config['model']['gru_decoder']['hidden_dim'] * (2 if self.config['model']['gru_decoder']['bidirectional'] else 1)
        self.classifier = nn.Linear(gru_out_dim, self.config['model']['num_classes'])
    

    def _apply_patching(self, x):
        """
        Apply patching to input
        x: (batch_size, time, input_dim)

        This chunks the time dimension into num_patches 
        overlapping patches of size patch_size. Then squashes
        the patch_size and feature dim back together. 
        Essentially this creates sliding window across time steps, 
        since 20ms bins (default data) is too small to capture
        phonemes.

        Returns: (batch_size, num_patches, patch_size * input_dim)
        """
        x = x.unsqueeze(1)                      # [batches, 1, timesteps, feature_dim]
        x = x.permute(0, 3, 1, 2)               # [batches, feature_dim, 1, timesteps]

        # Extract patches using unfold (sliding window)
        x_unfold = x.unfold(3, self.patch_size, self.patch_stride)  # [batches, feature_dim, 1, num_patches, patch_size]

        # Remove dummy height dimension and rearrange dimensions
        x_unfold = x_unfold.squeeze(2)           # [batches, feature_dum, num_patches, patch_size]
        x_unfold = x_unfold.permute(0, 2, 3, 1)  # [batches, num_patches, patch_size, feature_dim]

        # Flatten last two dimensions (patch_size and features)
        x = x_unfold.reshape(x.size(0), x_unfold.size(1), -1) 
        return x

    def forward(self, x, day_idx):
        """
        Forward pass through the model
        1. Pass input through day-specific adapter
        2. Apply patching to input
        3. Pass sequence through GRU decoder
        4. Pass GRU output through classifier
        """
        x = self.day_adapter(x, day_idx)
        x = self._apply_patching(x)
        x = self.gru_decoder(x)
        logits = self.classifier(x)
        return logits
