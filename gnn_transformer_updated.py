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
    Combined GNN + Transformer model with separate spatial and temporal embeddings
    """
    
    def __init__(self, 
                 node_feature_dim=2,
                 gnn_hidden_dim=64,
                 gnn_output_dim=128,
                 transformer_heads=4,
                 transformer_layers=4,
                 num_classes=2,
                 dropout=0.2,
                 max_num_keypoints=5):  # Added parameter for max keypoints
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
            max_num_keypoints: Maximum number of keypoints to consider
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
        self.max_num_keypoints = max_num_keypoints
        
        # GNN encoder for spatial information
        self.gnn_encoder = GNNEncoder(
            input_dim=node_feature_dim,
            hidden_dim=gnn_hidden_dim,
            output_dim=gnn_output_dim,
            dropout=dropout
        )
        
        # Transformer encoder for processing GNN embeddings (original pathway)
        self.transformer_encoder = TransformerEncoder(
            input_dim=gnn_output_dim,
            num_heads=transformer_heads,
            num_layers=transformer_layers,
            dropout=dropout
        )
        
        # === NEW COMPONENTS FOR DUAL PATHWAY ===
        # Projection layer for raw keypoints
        raw_keypoint_dim = max_num_keypoints * node_feature_dim
        self.temporal_projection = nn.Sequential(
            nn.Linear(raw_keypoint_dim, gnn_output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Separate transformer for temporal pathway
        self.temporal_transformer = TransformerEncoder(
            input_dim=gnn_output_dim,
            num_heads=transformer_heads,
            num_layers=transformer_layers,
            dropout=dropout
        )
        
        # Output projection for temporal pathway
        self.temporal_output = nn.Linear(gnn_output_dim, gnn_output_dim)
        
        # Fusion layer to combine spatial and temporal embeddings
        self.fusion_layer = nn.Sequential(
            nn.Linear(gnn_output_dim * 2, gnn_output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Classification head
        self.classification_head = nn.Sequential(
            nn.Linear(gnn_output_dim, gnn_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gnn_output_dim // 2, num_classes)
        )
    
    def forward(self, graphs, seq_lengths=None, raw_keypoints=None):
        """
        Forward pass with dual pathways
        
        Args:
            graphs: List of PyTorch Geometric Data objects for each frame
                [batch_size, seq_len] where each element is a Data object
            seq_lengths: Sequence lengths for each batch (batch_size)
            raw_keypoints: Raw keypoint sequences (batch_size, seq_len, num_keypoints, 2)
            
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
            mask = torch.ones(batch_size, max_seq_len, dtype=torch.bool, device=graphs[0][0].x.device)
        
        device = graphs[0][0].x.device
        
        # Check if we should use dual pathway or original approach
        use_dual_pathway = raw_keypoints is not None
        
        if use_dual_pathway:
            # === DUAL PATHWAY APPROACH ===
            
            # ===== SPATIAL PATHWAY (GNN) =====
            # Process all graphs for all frames to get graph-level embedding
            all_graphs = []
            for i in range(batch_size):
                all_graphs.extend(graphs[i])
            
            # Combine all graphs into a single batch
            all_batch = Batch.from_data_list(all_graphs)
            
            # Apply GNN encoder to all graphs at once
            all_graph_embeddings = self.gnn_encoder(all_batch.x, all_batch.edge_index, all_batch.batch)
            
            # Split the embeddings back by sequence
            offset = 0
            graph_embeddings_by_seq = []
            for i in range(batch_size):
                seq_len = len(graphs[i])
                seq_embeddings = all_graph_embeddings[offset:offset+seq_len]
                
                # Use only valid frames for pooling if seq_lengths provided
                if seq_lengths is not None:
                    valid_len = min(seq_lengths[i], seq_len)
                    spatial_embedding = torch.mean(seq_embeddings[:valid_len], dim=0, keepdim=True)
                else:
                    spatial_embedding = torch.mean(seq_embeddings, dim=0, keepdim=True)
                
                graph_embeddings_by_seq.append(spatial_embedding)
                offset += seq_len
            
            # Stack spatial embeddings
            spatial_embedding = torch.cat(graph_embeddings_by_seq, dim=0)  # (batch_size, gnn_output_dim)
            
            # ===== TEMPORAL PATHWAY (Transformer with raw keypoints) =====
            # Reshape and ensure we don't exceed max keypoints
            B, S, K, F = raw_keypoints.shape
            # Take up to max_num_keypoints
            K_used = min(K, self.max_num_keypoints)
            
            # Flatten keypoints for each frame
            keypoints_used = raw_keypoints[:, :, :K_used, :]
            temporal_input = keypoints_used.reshape(B, S, K_used * F)
            
            # Project to transformer dimension
            temporal_input = self.temporal_projection(temporal_input)
            
            # Apply transformer
            temporal_output = self.temporal_transformer(temporal_input, mask)
            
            # Global temporal pooling with mask
            mask_expanded = mask.unsqueeze(-1).expand_as(temporal_output)
            sum_temporal = torch.sum(temporal_output * mask_expanded, dim=1)
            sum_mask = torch.sum(mask_expanded, dim=1)
            temporal_embedding = sum_temporal / (sum_mask + 1e-10)
            
            # Final projection
            temporal_embedding = self.temporal_output(temporal_embedding)
            
        else:
            # === ORIGINAL APPROACH ===
            # Process each frame with GNN encoder
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
            
            gnn_embeddings = torch.stack(gnn_embeddings, dim=1)  # (batch_size, seq_len, gnn_output_dim)
            
            # Extract spatial information by averaging across time (only valid frames)
            mask_expanded = mask.unsqueeze(-1).expand_as(gnn_embeddings)
            sum_spatial = torch.sum(gnn_embeddings * mask_expanded, dim=1)
            sum_mask = torch.sum(mask_expanded, dim=1)
            spatial_embedding = sum_spatial / (sum_mask + 1e-10)  # (batch_size, gnn_output_dim)
            
            # Apply transformer encoder to extract temporal information
            transformer_output = self.transformer_encoder(gnn_embeddings, mask)  # (batch_size, seq_len, gnn_output_dim)
            
            # Extract temporal embedding with masked mean pooling
            sum_temporal = torch.sum(transformer_output * mask_expanded, dim=1)
            temporal_embedding = sum_temporal / (sum_mask + 1e-10)  # (batch_size, gnn_output_dim)
        
        # Combine spatial and temporal embeddings (same for both approaches)
        combined_embedding = torch.cat([spatial_embedding, temporal_embedding], dim=1)  # (batch_size, gnn_output_dim*2)
        
        # Apply fusion layer
        fused_embedding = self.fusion_layer(combined_embedding)  # (batch_size, gnn_output_dim)
        
        # Apply classification head
        logits = self.classification_head(fused_embedding)
        
        return logits
    
    def get_embeddings(self, graphs, seq_lengths=None, raw_keypoints=None):
        """
        Get spatial and temporal embeddings for analysis
        
        Args:
            graphs: List of PyTorch Geometric Data objects for each frame
            seq_lengths: Sequence lengths for each batch
            raw_keypoints: Raw keypoint sequences (optional)
            
        Returns:
            Tuple of (spatial_embedding, temporal_embedding, fused_embedding)
        """
        # Create forward pass with no gradient tracking
        with torch.no_grad():
            batch_size = len(graphs)
            max_seq_len = max(len(seq) for seq in graphs)
            
            # Create a mask for valid frames
            if seq_lengths is not None:
                mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool, device=graphs[0][0].x.device)
                for i, length in enumerate(seq_lengths):
                    mask[i, :length] = 1
            else:
                mask = torch.ones(batch_size, max_seq_len, dtype=torch.bool, device=graphs[0][0].x.device)
            
            device = graphs[0][0].x.device
            
            # Check if we should use dual pathway or original approach
            use_dual_pathway = raw_keypoints is not None
            
            if use_dual_pathway:
                # === DUAL PATHWAY APPROACH ===
                
                # ===== SPATIAL PATHWAY (GNN) =====
                all_graphs = []
                for i in range(batch_size):
                    all_graphs.extend(graphs[i])
                
                all_batch = Batch.from_data_list(all_graphs)
                all_graph_embeddings = self.gnn_encoder(all_batch.x, all_batch.edge_index, all_batch.batch)
                
                offset = 0
                graph_embeddings_by_seq = []
                for i in range(batch_size):
                    seq_len = len(graphs[i])
                    seq_embeddings = all_graph_embeddings[offset:offset+seq_len]
                    
                    if seq_lengths is not None:
                        valid_len = min(seq_lengths[i], seq_len)
                        spatial_embedding = torch.mean(seq_embeddings[:valid_len], dim=0, keepdim=True)
                    else:
                        spatial_embedding = torch.mean(seq_embeddings, dim=0, keepdim=True)
                    
                    graph_embeddings_by_seq.append(spatial_embedding)
                    offset += seq_len
                
                spatial_embedding = torch.cat(graph_embeddings_by_seq, dim=0)
                
                # ===== TEMPORAL PATHWAY (Transformer with raw keypoints) =====
                B, S, K, F = raw_keypoints.shape
                K_used = min(K, self.max_num_keypoints)
                
                keypoints_used = raw_keypoints[:, :, :K_used, :]
                temporal_input = keypoints_used.reshape(B, S, K_used * F)
                
                temporal_input = self.temporal_projection(temporal_input)
                temporal_output = self.temporal_transformer(temporal_input, mask)
                
                mask_expanded = mask.unsqueeze(-1).expand_as(temporal_output)
                sum_temporal = torch.sum(temporal_output * mask_expanded, dim=1)
                sum_mask = torch.sum(mask_expanded, dim=1)
                temporal_embedding = sum_temporal / (sum_mask + 1e-10)
                
                temporal_embedding = self.temporal_output(temporal_embedding)
                
            else:
                # === ORIGINAL APPROACH ===
                # Same as before
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
                
                # Extract spatial embedding
                mask_expanded = mask.unsqueeze(-1).expand_as(gnn_embeddings)
                sum_spatial = torch.sum(gnn_embeddings * mask_expanded, dim=1)
                sum_mask = torch.sum(mask_expanded, dim=1)
                spatial_embedding = sum_spatial / (sum_mask + 1e-10)
                
                # Extract temporal embedding
                transformer_output = self.transformer_encoder(gnn_embeddings, mask)
                sum_temporal = torch.sum(transformer_output * mask_expanded, dim=1)
                temporal_embedding = sum_temporal / (sum_mask + 1e-10)
            
            # Combine embeddings
            combined_embedding = torch.cat([spatial_embedding, temporal_embedding], dim=1)
            fused_embedding = self.fusion_layer(combined_embedding)
        
        return spatial_embedding, temporal_embedding, fused_embedding