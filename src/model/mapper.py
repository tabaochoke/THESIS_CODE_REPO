import torch.nn as nn
from config_loader import ConfigLoader

class MapperModel(nn.Module):
    def __init__(self, in_embedding_dim=ConfigLoader.get("model.in_embedding_dim"), out_embedding_dim = ConfigLoader.get("model.out_embedding_dim"),
                 hidden_dim=ConfigLoader.get("model.mapper_hidden_dim"), num_heads=ConfigLoader.get("model.mapper_num_heads"), 
                 num_layers=ConfigLoader.get("model.mapper_num_layers"), dropout=ConfigLoader.get("model.mapper_dropout")):
        """
        A more advanced model for mapping embeddings using transformer-based architecture.
        
        Args:
            in_embedding_dim (int): Dimension of input embeddings (Vietnamese embeddings).
            out_embedding_dim (int): Dimension of output embeddings (English embeddings).
            hidden_dim (int): Dimension of the hidden layers in the transformer.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of transformer encoder layers.
            dropout (float): Dropout rate.
        """
        super(MapperModel, self).__init__()
        
        # Project input embeddings to hidden_dim
        self.input_projection = nn.Linear(in_embedding_dim, hidden_dim)
        self.input_layernorm = nn.LayerNorm(hidden_dim)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.encoder_layernorm = nn.LayerNorm(hidden_dim)
        
        # Output projection to map to out_embedding_dim
        self.output_projection = nn.Linear(hidden_dim, out_embedding_dim)
        self.output_layernorm = nn.LayerNorm(out_embedding_dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, vietnamese_embedding):
        """
        Forward pass for mapping embeddings.
        Args:
            vietnamese_embedding (Tensor): Input embeddings of shape (batch_size, seq_len, in_embedding_dim).
        
        Returns:
            Tensor: Output embeddings of shape (batch_size, seq_len, out_embedding_dim).
        """
        # Project input embeddings
        x = self.input_projection(vietnamese_embedding)  # (batch_size, seq_len, hidden_dim)
        x = self.input_layernorm(x)  # Normalize after projection
        
        # Prepare input for transformer (seq_len, batch_size, hidden_dim)
        x = x.permute(1, 0, 2)
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x)  # (seq_len, batch_size, hidden_dim)
        x = self.encoder_layernorm(x)  # Normalize after encoder
        
        # Revert back to (batch_size, seq_len, hidden_dim)
        x = x.permute(1, 0, 2)
        
        # Project to output embedding space
        x = self.output_projection(x)  # (batch_size, seq_len, out_embedding_dim)
        x = self.output_layernorm(x)  # Normalize output
        
        # Apply dropout
        x = self.dropout(x)
        
        return x
