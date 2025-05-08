import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
import numpy as np


class GNNEncoder(nn.Module):
    """
    Graph Neural Network encoder for processing keypoint data
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        """
        Initialize the GNN encoder
        
        Args:
            input_dim: Input dimension of node features
            hidden_dim: Hidden dimension of GNN layers
            output_dim: Output dimension of GNN embedding
            dropout: Dropout rate
        """
        super(GNNEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        
        # GNN layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        
        # Normalization and dropout
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, batch):
        """
        Forward pass
        
        Args:
            x: Node features (num_nodes, input_dim)
            edge_index: Graph connectivity (2, num_edges)
            batch: Batch assignment (num_nodes)
            
        Returns:
            Graph-level embedding (batch_size, output_dim)
        """
        # First GNN layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.norm1(x)
        x = self.dropout_layer(x)
        
        # Second GNN layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.norm2(x)
        x = self.dropout_layer(x)
        
        # Third GNN layer
        x = self.conv3(x, edge_index)
        
        # Global pooling to get graph-level embeddings
        x = torch.cat([
            torch.mean(x[batch == i], dim=0, keepdim=True) 
            for i in range(batch.max().item() + 1)
        ], dim=0)
        
        return x


class TransformerEncoder(nn.Module):
    """
    Transformer encoder for processing temporal sequences of GNN embeddings
    """
    
    def __init__(self, input_dim, num_heads=4, num_layers=4, dropout=0.2):
        """
        Initialize the transformer encoder
        
        Args:
            input_dim: Input dimension of GNN embeddings
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
        """
        super(TransformerEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=4 * input_dim,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        
        # Transformer encoder
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
        # Position encoding
        self.position_encoding = PositionalEncoding(
            d_model=input_dim,
            dropout=dropout,
            max_len=1000  # Maximum sequence length
        )
    
    def forward(self, x, mask=None):
        """
        Forward pass
        
        Args:
            x: Sequence of GNN embeddings (batch_size, seq_len, input_dim)
            mask: Attention mask (batch_size, seq_len)
            
        Returns:
            Sequence encoding (batch_size, seq_len, input_dim)
        """
        # Add positional encoding
        x = self.position_encoding(x)
        
        # Create transformer attention mask
        if mask is not None:
            # Convert boolean mask to attention mask
            attention_mask = torch.zeros(
                mask.size(0), mask.size(1), mask.size(1), 
                device=mask.device
            )
            for i in range(mask.size(0)):
                valid_len = mask[i].sum().item()
                attention_mask[i, :valid_len, :valid_len] = 1
            
            # Apply the attention mask
            x = self.transformer(x, src_key_padding_mask=~mask)
        else:
            # No mask
            x = self.transformer(x)
        
        return x


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models
    """
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Initialize positional encoding
        
        Args:
            d_model: Embedding dimension
            dropout: Dropout rate
            max_len: Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register buffer (not a parameter)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            
        Returns:
            Tensor with positional encoding added (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class GNNTransformerModel(nn.Module):
    """
    Combined GNN + Transformer model for keypoint-based score generation
    """
    
    def __init__(self, 
                 node_feature_dim=2,
                 gnn_hidden_dim=64,
                 gnn_output_dim=128,
                 transformer_heads=4,
                 transformer_layers=4,
                 num_classes=2,
                 dropout=0.2):
        """
        Initialize the GNN + Transformer model
        
        Args:
            node_feature_dim: Dimension of node features (default is 2 for x, y coordinates)
            gnn_hidden_dim: Hidden dimension of GNN layers
            gnn_output_dim: Output dimension of GNN embedding
            transformer_heads: Number of transformer attention heads
            transformer_layers: Number of transformer layers
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super(GNNTransformerModel, self).__init__()
        
        # Model parameters
        self.node_feature_dim = node_feature_dim
        self.gnn_hidden_dim = gnn_hidden_dim
        self.gnn_output_dim = gnn_output_dim
        self.transformer_heads = transformer_heads
        self.transformer_layers = transformer_layers
        self.num_classes = num_classes
        self.dropout = dropout
        
        # GNN encoder for spatial information
        self.gnn_encoder = GNNEncoder(
            input_dim=node_feature_dim,
            hidden_dim=gnn_hidden_dim,
            output_dim=gnn_output_dim,
            dropout=dropout
        )
        
        # Transformer encoder for temporal information
        self.transformer_encoder = TransformerEncoder(
            input_dim=gnn_output_dim,
            num_heads=transformer_heads,
            num_layers=transformer_layers,
            dropout=dropout
        )
        
        # Classification head
        self.classification_head = nn.Sequential(
            nn.Linear(gnn_output_dim, gnn_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gnn_output_dim // 2, num_classes)
        )
    
    def forward(self, graphs, seq_lengths=None):
        """
        Forward pass
        
        Args:
            graphs: List of PyTorch Geometric Data objects for each frame
                   [batch_size, seq_len] where each element is a Data object
            seq_lengths: Sequence lengths for each batch (batch_size)
            
        Returns:
            Class logits (batch_size, num_classes)
        """
        batch_size = len(graphs)
        max_seq_len = max(len(seq) for seq in graphs)
        
        # Create a mask for valid frames
        if seq_lengths is not None:
            mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool, device=graphs[0][0].x.device)
            for i, length in enumerate(seq_lengths):
                mask[i, :length] = 1
        else:
            mask = None
        
        # Process each frame with GNN encoder
        gnn_embeddings = []
        for t in range(max_seq_len):
            # Collect graphs for this time step
            frame_graphs = []
            for i in range(batch_size):
                if t < len(graphs[i]):
                    frame_graphs.append(graphs[i][t])
                else:
                    # Padding with empty graph with same device
                    device = graphs[i][0].x.device
                    # Create a dummy single-node graph for padding
                    dummy_graph = Data(
                        x=torch.zeros(1, self.node_feature_dim, device=device),
                        edge_index=torch.zeros(2, 0, dtype=torch.long, device=device)
                    )
                    frame_graphs.append(dummy_graph)
            
            # Combine graphs into a batch
            batch = Batch.from_data_list(frame_graphs)
            
            # Apply GNN encoder
            embeddings = self.gnn_encoder(batch.x, batch.edge_index, batch.batch)
            gnn_embeddings.append(embeddings)
        
        # Stack embeddings to form a sequence
        gnn_embeddings = torch.stack(gnn_embeddings, dim=1)  # (batch_size, seq_len, gnn_output_dim)
        
        # Apply transformer encoder
        transformer_output = self.transformer_encoder(gnn_embeddings, mask)
        
        # Global temporal pooling
        if mask is not None:
            # Masked mean pooling
            mask_expanded = mask.unsqueeze(-1).expand_as(transformer_output)
            sum_embeddings = torch.sum(transformer_output * mask_expanded, dim=1)
            sum_mask = torch.sum(mask_expanded, dim=1)
            pooled_output = sum_embeddings / (sum_mask + 1e-10)
        else:
            # Simple mean pooling
            pooled_output = torch.mean(transformer_output, dim=1)
        
        # Apply classification head
        logits = self.classification_head(pooled_output)
        
        return logits
    
    def get_attention_weights(self, graphs, seq_lengths=None):
        """
        Get attention weights for visualization
        
        Args:
            graphs: List of PyTorch Geometric Data objects for each frame
            seq_lengths: Sequence lengths for each batch
            
        Returns:
            Attention weights from transformer layers
        """
        batch_size = len(graphs)
        max_seq_len = max(len(seq) for seq in graphs)
        
        # Create a mask for valid frames
        if seq_lengths is not None:
            mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool, device=graphs[0][0].x.device)
            for i, length in enumerate(seq_lengths):
                mask[i, :length] = 1
        else:
            mask = None
        
        # Process each frame with GNN encoder (same as forward)
        gnn_embeddings = []
        for t in range(max_seq_len):
            frame_graphs = []
            for i in range(batch_size):
                if t < len(graphs[i]):
                    frame_graphs.append(graphs[i][t])
                else:
                    device = graphs[i][0].x.device
                    dummy_graph = Data(
                        x=torch.zeros(1, self.node_feature_dim, device=device),
                        edge_index=torch.zeros(2, 0, dtype=torch.long, device=device)
                    )
                    frame_graphs.append(dummy_graph)
            
            batch = Batch.from_data_list(frame_graphs)
            embeddings = self.gnn_encoder(batch.x, batch.edge_index, batch.batch)
            gnn_embeddings.append(embeddings)
        
        gnn_embeddings = torch.stack(gnn_embeddings, dim=1)
        
        # Add positional encoding
        x = self.transformer_encoder.position_encoding(gnn_embeddings)
        
        # Get attention weights from each transformer layer
        attention_weights = []
        
        # Access transformer layers
        for layer in self.transformer_encoder.transformer.layers:
            # Forward through self-attention
            attn_output, attn_weights = layer.self_attn(x, x, x, attn_mask=None, key_padding_mask=~mask if mask is not None else None, need_weights=True)
            attention_weights.append(attn_weights.detach())
            
            # Continue forward through the rest of the layer (similar to original implementation)
            attn_output = layer.dropout1(attn_output)
            x = x + attn_output
            x = layer.norm1(x)
            ff_output = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
            ff_output = layer.dropout2(ff_output)
            x = x + ff_output
            x = layer.norm2(x)
        
        return attention_weights