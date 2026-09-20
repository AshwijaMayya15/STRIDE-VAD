import torch
from torch import nn
from utils import FeedForward, DECOUPLED
from performer_pytorch import Performer


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool3d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (B, C, T, H, W)
        b, c = x.shape[:2]
        # Squeeze: Global average pooling
        y = self.squeeze(x).view(b, c)
        # Excitation: FC layers
        y = self.excitation(y).view(b, c, 1, 1, 1)
        # Scale
        return x * y.expand_as(x)


class ConvLSTMCell(nn.Module):
    """Convolutional LSTM cell for spatial-temporal processing"""
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        
        # Combined convolution for all gates: input, forget, cell, output
        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size,
            padding=padding,
            bias=True
        )
        
    def forward(self, x, hidden):
        """
        x: (B, C, H, W)
        hidden: tuple of (h, c) where each is (B, C, H, W)
        """
        h, c = hidden
        combined = torch.cat([x, h], dim=1)  # Concatenate input and hidden state
        gates = self.conv(combined)
        
        # Split into 4 gates
        i, f, o, g = torch.split(gates, self.hidden_dim, dim=1)
        
        # Apply activations
        i = torch.sigmoid(i)  # Input gate
        f = torch.sigmoid(f)  # Forget gate
        o = torch.sigmoid(o)  # Output gate
        g = torch.tanh(g)     # Cell gate
        
        # Compute new cell and hidden state
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next


class ConvLSTMBlock(nn.Module):
    """ConvLSTM block that processes temporal sequences with spatial awareness"""
    def __init__(self, dim, heads=16, kernel_size=3):
        super().__init__()
        self.dim = dim
        self.cell = ConvLSTMCell(dim, dim, kernel_size=kernel_size)
        
    def forward(self, x):
        """
        x shape: (B, T, H, W, C)
        Returns: (B, T, H, W, C)
        """
        B, T, H, W, C = x.shape
        
        # Initialize hidden states (zeros)
        device = x.device
        h = torch.zeros(B, C, H, W, device=device)
        c = torch.zeros(B, C, H, W, device=device)
        
        outputs = []
        for t in range(T):
            # Get frame at time t and permute to (B, C, H, W)
            x_t = x[:, t].permute(0, 3, 1, 2)
            
            # Apply ConvLSTM cell
            h, c = self.cell(x_t, (h, c))
            
            # Permute back to (B, H, W, C) and store
            outputs.append(h.permute(0, 2, 3, 1))
        
        # Stack outputs along time dimension
        output = torch.stack(outputs, dim=1)  # (B, T, H, W, C)
        return output


class DECOUPLED_LSTM(nn.Module):
    """
    Hybrid: Spatial Conv2D + Temporal LSTM
    Replaces Conv1D in DECOUPLED with LSTM for better temporal modeling
    """
    def __init__(self, dim, heads=16, lstm_hidden=None):
        super().__init__()
        self.dim = dim
        lstm_hidden = lstm_hidden or (dim // 2)
        
        # Spatial processing with Conv2D (efficient, parallel)
        self.spatial_conv = nn.Conv2d(
            dim, dim, 
            kernel_size=3, 
            padding=1, 
            groups=heads,  # Depthwise
            bias=False
        )
        self.spatial_norm = nn.BatchNorm2d(dim)
        
        # Temporal processing with LSTM (replaces Conv1D)
        self.temporal_lstm = nn.LSTM(
            input_size=dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Project back to original dim (hidden*2 because bidirectional)
        self.temporal_proj = nn.Linear(lstm_hidden * 2, dim)
        self.temporal_norm = nn.LayerNorm(dim)
        
        self.activation = nn.GELU()
        
    def forward(self, x):
        """
        x shape: (B, T, H, W, C)
        """
        B, T, H, W, C = x.shape
        
        # 1. SPATIAL: Apply Conv2D per frame
        x_spatial = []
        for t in range(T):
            # Get frame at time t: (B, H, W, C) -> (B, C, H, W)
            frame = x[:, t].permute(0, 3, 1, 2)
            
            # Spatial conv + norm + activation
            frame = self.spatial_conv(frame)
            frame = self.spatial_norm(frame)
            frame = self.activation(frame)
            
            # Back to (B, H, W, C)
            x_spatial.append(frame.permute(0, 2, 3, 1))
        
        x = torch.stack(x_spatial, dim=1)  # (B, T, H, W, C)
        
        # 2. TEMPORAL: Apply LSTM per spatial location
        # Reshape: (B, T, H, W, C) -> (B*H*W, T, C)
        x = x.permute(0, 2, 3, 1, 4).reshape(B * H * W, T, C)
        
        # Temporal LSTM
        x, _ = self.temporal_lstm(x)  # (B*H*W, T, hidden*2)
        x = self.temporal_proj(x)     # (B*H*W, T, C)
        x = self.temporal_norm(x)
        
        # Reshape back: (B*H*W, T, C) -> (B, T, H, W, C)
        x = x.view(B, H, W, T, C).permute(0, 3, 1, 2, 4)
        
        return x


class TemporalLSTM(nn.Module):
    """Bidirectional LSTM for temporal modeling between Conv and Attention stages"""
    def __init__(self, dim, hidden_dim=None, num_layers=2, dropout=0.3):
        super().__init__()
        hidden_dim = hidden_dim or (dim // 2)  # Default: hidden=64 for dim=128
        
        self.lstm = nn.LSTM(
            input_size=dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Project back to original dimension (hidden*2 because bidirectional)
        self.projection = nn.Linear(hidden_dim * 2, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Input x shape: (B, T, H, W, C)
        Process each spatial location's temporal sequence
        """
        B, T, H, W, C = x.shape
        
        # Reshape to process all spatial locations independently
        # (B, T, H, W, C) -> (B*H*W, T, C)
        x = x.permute(0, 2, 3, 1, 4).contiguous()  # (B, H, W, T, C)
        x = x.view(B * H * W, T, C)
        
        # Apply LSTM
        lstm_out, _ = self.lstm(x)  # (B*H*W, T, hidden*2)
        
        # Project back to original dimension
        x = self.projection(lstm_out)  # (B*H*W, T, C)
        
        # Reshape back to original spatial-temporal structure
        x = x.view(B, H, W, T, C)
        x = x.permute(0, 3, 1, 2, 4).contiguous()  # (B, T, H, W, C)
        
        # Normalize and dropout
        x = self.norm(x)
        x = self.dropout(x)
        
        return x


class TemporalPyramidPool(nn.Module):
    """Multi-scale temporal feature extraction"""
    def __init__(self, in_channels):
        super().__init__()
        self.pools = nn.ModuleList([
            nn.AdaptiveMaxPool3d((t, 1, 1)) for t in [1, 2, 4, 8]
        ])
        # Each pool outputs (B, C, t, 1, 1) which flattens to (B, C*t)
        # Total: C*1 + C*2 + C*4 + C*8 = C*15
        self.projection = nn.Linear(in_channels * 15, in_channels)
        
    def forward(self, x):
        # x shape: (B, C, T, H, W)
        pooled = [pool(x).flatten(1) for pool in self.pools]
        concatenated = torch.cat(pooled, dim=1)
        return self.projection(concatenated)


class AttnBlock(nn.Module):
    def __init__(self, dim, depth, dropout, attn_dropout, heads=16, ff_mult=2):
        super().__init__()
        self.performer = Performer(
            dim=dim, 
            depth=depth, 
            heads=heads, 
            dim_head=dim // heads, 
            causal=False,
            ff_mult=ff_mult,
            local_attn_heads=8,
            local_window_size=dim // 8,
            ff_dropout=dropout,
            attn_dropout=attn_dropout,
        )

    def forward(self, x):
        B, T, H, W, C = x.shape
        x = x.view(B, -1, C)
        x = self.performer(x)
        x = x.view(B, T, H, W, C)
        return x


class ConvBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        ff_mult=2,
        dropout=0.,
        heads=16,
        use_convlstm=False,
        use_decoupled_lstm=False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Choose convolution type
        if use_convlstm:
            self.conv = ConvLSTMBlock(dim, heads)  # Full ConvLSTM
        elif use_decoupled_lstm:
            self.conv = DECOUPLED_LSTM(dim, heads)  # Hybrid: Conv2D + LSTM
        else:
            self.conv = DECOUPLED(dim, heads)       # Original DECOUPLED
        
        self.ff = FeedForward(dim, ff_mult, dropout)

    def forward(self, x):
        x = x + self.conv(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class EnhancedModel(nn.Module):
    """
    Enhanced model with:
    - Deeper architecture (4 stages)
    - SE blocks for channel attention (optional)
    - Temporal pyramid pooling (optional)
    - Bidirectional LSTM for temporal modeling (optional)
    - ConvLSTM or Hybrid DECOUPLED_LSTM in conv blocks (optional)
    - Residual connections
    - Better regularization
    """
    def __init__(
        self,
        *,
        dropout=0.3,
        attn_dropout=0.1,
        ff_mult=4,
        dims=(192, 192, 128, 128),  # 4-stage architecture
        depths=(4, 3, 3, 2),  # Deeper blocks
        block_types=('c', 'c', 'a', 'a'),
        input_channels=198,  # 192 for X3D only, 198 for X3D+Flow fusion
        use_se=True,  # Use SE blocks
        use_tpp=True,  # Use Temporal Pyramid Pooling
        use_lstm=True,  # Use BiLSTM between stages
        lstm_hidden=64,  # LSTM hidden dimension
        lstm_layers=2,   # Number of LSTM layers
        use_convlstm=False,  # Use full ConvLSTM in conv blocks
        use_decoupled_lstm=False,  # Use hybrid DECOUPLED_LSTM in conv blocks
    ):
        super().__init__()
        self.init_dim, *_, last_dim = dims
        self.input_channels = input_channels
        self.use_se = use_se
        self.use_tpp = use_tpp
        self.use_lstm = use_lstm
        self.use_convlstm = use_convlstm
        self.use_decoupled_lstm = use_decoupled_lstm

        # Input projection
        self.norm0 = nn.LayerNorm(input_channels)
        self.linear = nn.Linear(input_channels, dims[0])
        
        # Build stages
        self.stages = nn.ModuleList([])
        self.se_blocks = nn.ModuleList([]) if use_se else None
        self.lstm_layer = None  # Will be added between conv and attn stages

        for ind, (depth, block_type) in enumerate(zip(depths, block_types)):
            is_last = ind == len(depths) - 1
            stage_dim = dims[ind]
            
            # Add convolutional or attention blocks
            if block_type == "c":
                for _ in range(depth):
                    self.stages.append(
                        ConvBlock(
                            dim=stage_dim,
                            ff_mult=ff_mult,
                            dropout=dropout,
                            use_convlstm=use_convlstm,
                            use_decoupled_lstm=use_decoupled_lstm,
                        )
                    )
            elif block_type == "a":
                for _ in range(depth):
                    self.stages.append(
                        AttnBlock(stage_dim, 1, dropout, attn_dropout, ff_mult=ff_mult)
                    )
            
            # Add SE block after each stage
            if use_se:
                self.se_blocks.append(SEBlock(stage_dim, reduction=16))
            
            # Downsample/project to next stage
            if not is_last:
                self.stages.append(
                    nn.Sequential(
                        nn.LayerNorm(stage_dim),
                        nn.Linear(stage_dim, dims[ind + 1]),
                        nn.Dropout(dropout * 0.5)  # Light dropout in transitions
                    )
                )
        
        # Add LSTM after conv stages (after transition to 128-dim)
        if use_lstm:
            lstm_dim = dims[2]  # Dimension after conv stages (128)
            self.lstm_layer = TemporalLSTM(
                dim=lstm_dim,
                hidden_dim=lstm_hidden,
                num_layers=lstm_layers,
                dropout=dropout
            )

        # Multi-scale temporal pooling or regular pooling
        if use_tpp:
            self.temporal_pool = TemporalPyramidPool(last_dim)
            self.spatial_pool = nn.AdaptiveMaxPool2d((1, 1))
        else:
            self.pooling = nn.AdaptiveMaxPool3d((1, 1, 1))
        
        # Final classification layers
        self.norm = nn.LayerNorm(last_dim)
        self.drop_out = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(last_dim, last_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(last_dim // 2, 1)
        )
        
    def forward(self, x):
        """
        Forward pass for enhanced spatial-temporal model.
        
        Input x shape: (B, C, T, H, W)
        - C = 192 for X3D only
        - C = 198 for X3D + Flow fusion (192 + 6)
        """
        
        # Permute to (B, T, H, W, C) for processing
        x = x.permute(0, 2, 3, 4, 1)

        # Project input to model dimension
        if x.shape[4] != self.init_dim:
            x = self.linear(self.norm0(x))

        # Track stage for LSTM insertion
        lstm_applied = False
        
        # Process through all stages with SE blocks
        se_idx = 0
        stage_idx = 0
        for stage in self.stages:
            x = stage(x)
            
            # Insert LSTM right after transition to 128-dim (before first attention block)
            if self.use_lstm and not lstm_applied:
                # Check if this was a transition layer that brought us to 128-dim
                if isinstance(stage, nn.Sequential) and x.shape[-1] == 128:
                    x = self.lstm_layer(x)
                    lstm_applied = True
            
            # Apply SE block after each complete stage (not after transitions)
            if self.use_se and isinstance(stage, (ConvBlock, AttnBlock)):
                # Check if this is the last block of a stage
                # (next stage is a transition or we're at the end)
                is_stage_end = (stage_idx == len(self.stages) - 1 or 
                               isinstance(self.stages[stage_idx + 1], nn.Sequential))
                
                if is_stage_end and se_idx < len(self.se_blocks):
                    # Permute to (B, C, T, H, W) for SE block
                    x_se = x.permute(0, 4, 1, 2, 3)
                    x_se = self.se_blocks[se_idx](x_se)
                    x = x_se.permute(0, 2, 3, 4, 1)
                    se_idx += 1
            
            stage_idx += 1

        # Permute back to (B, C, T, H, W) for pooling
        x = x.permute(0, 4, 1, 2, 3)
        
        # Multi-scale pooling or regular pooling
        if self.use_tpp:
            # Temporal pyramid pooling + spatial pooling
            x_temporal = self.temporal_pool(x)  # (B, C)
            # Also do spatial pooling on full temporal sequence
            b, c, t, h, w = x.shape
            x_spatial = x.permute(0, 1, 3, 4, 2).reshape(b, c, h * w, t)
            x_spatial = x_spatial.mean(dim=3).reshape(b, c, h, w)
            x_spatial = self.spatial_pool(x_spatial).squeeze()  # (B, C)
            # Combine both
            x = (x_temporal + x_spatial) / 2
        else:
            x = self.pooling(x).squeeze()

        # Final classification
        x = self.drop_out(x)
        x = self.norm(x)
        logits = self.fc(x)
        
        return logits, x


# Alias for compatibility
Model = EnhancedModel