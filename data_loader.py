import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx

class KeypointDataset(Dataset):
    """
    Dataset for keypoint-based data from multiple modalities
    """
    
    def __init__(self, 
                segment_data, 
                view_type,
                seq_length=20,
                transform=None,
                include_hand=True,
                include_object=True):
        """
        Initialize the dataset
        
        Args:
            segment_data: Dictionary of segment data
            view_type: Camera view type ('top' or 'ipsi')
            seq_length: Maximum sequence length
            transform: Optional transform to apply to the data
            include_hand: Whether to include hand keypoints
            include_object: Whether to include object locations
        """
        self.segment_data = segment_data
        self.view_type = view_type
        self.seq_length = seq_length
        self.transform = transform
        self.include_hand = include_hand
        self.include_object = include_object
        
        # Filter segments by view type
        self.segment_ids = [
            sid for sid, segment in segment_data.items()
            if segment['view_type'] == view_type and segment['label'] is not None
        ]
        
        print(f"Loaded {len(self.segment_ids)} segments for {view_type} view")
    
    def __len__(self):
        """Get the number of segments"""
        return len(self.segment_ids)
    
    def __getitem__(self, idx):
        """
        Get a segment by index
        
        Args:
            idx: Segment index
            
        Returns:
            Tuple of (graphs, label, segment_id)
            - graphs: List of PyTorch Geometric Data objects for each frame
            - label: Segment label
            - segment_id: Segment ID
        """
        segment_id = self.segment_ids[idx]
        segment = self.segment_data[segment_id]
        
        # Get body keypoints
        body_keypoints = segment['body_keypoints']
        
        # Get hand keypoints if available
        hand_keypoints = None
        if self.include_hand and segment['hand_keypoints'] is not None:
            hand_keypoints = segment['hand_keypoints']
        
        # Get object locations if available
        object_locations = None
        if self.include_object and segment['object_locations'] is not None:
            object_locations = segment['object_locations']
        # Get label
        label = segment['label']

        # Determine which side is impaired based on video_id or another field
        impaired_side = segment['impaired_hand']  # Default   
        # Convert to graph representation
        graphs = self._create_graphs(body_keypoints, hand_keypoints, object_locations, impaired_side)
        
        # Apply transform if provided
        if self.transform:
            graphs = self.transform(graphs)
        
        return graphs, label, segment_id
    
    def _create_graphs(self, body_keypoints, hand_keypoints, object_locations, impaired_side='right'):
        """
        Create graph representations for each frame
        
        Args:
            body_keypoints: Body keypoints data
            hand_keypoints: Hand keypoints data (optional)
            object_locations: Object locations data (optional)
            impaired_side: Which side is impaired ('right' or 'left')
            
        Returns:
            List of PyTorch Geometric Data objects for each frame
        """
        # Determine the number of frames
        num_frames = min(body_keypoints.shape[2], self.seq_length)
        
        # Create graphs for each frame
        graphs = []
        for i in range(num_frames):
            # Get keypoints for this frame
            body_frame = body_keypoints[:, :, i]
            
            # Get hand keypoints for this frame (if available)
            hand_frame = None
            if hand_keypoints is not None:
                # Handle different formats based on the actual data
                if isinstance(hand_keypoints, np.ndarray) and hand_keypoints.ndim == 3:
                    if i < hand_keypoints.shape[2]:
                        hand_frame = hand_keypoints[:, :, i]
                elif isinstance(hand_keypoints, list) and i < len(hand_keypoints):
                    hand_frame = np.array(hand_keypoints[i])
            
            # Get object locations for this frame (if available)
            object_frame = None
            if object_locations is not None:
                # Handle different formats based on the actual data
                if isinstance(object_locations, np.ndarray) and object_locations.ndim == 3:
                    if i < object_locations.shape[2]:
                        object_frame = object_locations[:, :, i]
                elif isinstance(object_locations, list) and i < len(object_locations):
                    object_frame = np.array(object_locations[i])
            
            # Create graph for this frame
            graph = self._create_frame_graph(body_frame, hand_frame, object_frame, impaired_side)
            graphs.append(graph)
        
        return graphs
    
    def _create_frame_graph(self, body_keypoints, hand_keypoints, object_locations, impaired_side='right'):
        """
        Create a graph representation for a single frame, focusing on key joints
        
        Args:
            body_keypoints: Body keypoints for this frame
            hand_keypoints: Hand keypoints for this frame (optional)
            object_locations: Object locations for this frame (optional)
            impaired_side: Which side is impaired ('right' or 'left')
            
        Returns:
            PyTorch Geometric Data object
        """
        # Define key body joint indices for OpenPose format
        # OpenPose keypoint order:
        # 0: Nose, 1: Neck, 2: RShoulder, 3: RElbow, 4: RWrist, 5: LShoulder, 6: LElbow, 7: LWrist
        # 8: MidHip, 9-13: RHip-LAnkle, 14-18: REye-REar-LEar
        
        # Select the impaired side indices
        if impaired_side == 'right':
            # Right arm is impaired: Neck(1) -> RShoulder(2) -> RElbow(3) -> RWrist(4)
            body_key_indices = [1, 2, 3, 4, 8]  # Neck, RShoulder, RElbow, RWrist, MidHip
            body_wrist_index = 4  # Right wrist
            body_edges = [(0, 1), (1, 2), (2, 3), (0, 4)]  # Neck-Shoulder-Elbow-Wrist, Neck-MidHip
        else:
            # Left arm is impaired: Neck(1) -> LShoulder(5) -> LElbow(6) -> LWrist(7)
            body_key_indices = [1, 5, 6, 7, 8]  # Neck, LShoulder, LElbow, LWrist, MidHip
            body_wrist_index = 7  # Left wrist
            body_edges = [(0, 1), (1, 2), (2, 3), (0, 4)]  # Neck-Shoulder-Elbow-Wrist, Neck-MidHip
        
        # Define key hand joint indices for MediaPipe format
        # MediaPipe hand landmark model has 21 points:
        # 0: Wrist
        # 4: Thumb tip, 8: Index tip, 12: Middle tip, 16: Ring tip, 20: Pinky tip
        hand_key_indices = [0, 4, 8, 12, 16, 20]  # Wrist and fingertips
        
        # Create hand edges (connect wrist to each fingertip)
        hand_edges = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]  # Wrist-to-fingertips
        
        # Process keypoints
        x_list = []
        edge_index_list = []
        node_offset = 0
        
        # Add selected body keypoints
        if body_keypoints is not None and body_keypoints.size > 0:
            # Extract only the key body joints
            key_body_points = body_keypoints[body_key_indices]
            
            # Convert to tensor
            body_tensor = torch.tensor(key_body_points, dtype=torch.float)
            x_list.append(body_tensor)
            
            # Create body edges
            for i, (src, dst) in enumerate(body_edges):
                edge_index_list.append((src, dst))
                edge_index_list.append((dst, src))  # Bidirectional
            
            node_offset += body_tensor.shape[0]
        
        # Add selected hand keypoints
        if hand_keypoints is not None and (isinstance(hand_keypoints, np.ndarray) and hand_keypoints.size > 0):
            # Extract only the key hand joints (wrist and fingertips)
            if hand_keypoints.shape[0] >= 21:  # Ensure we have enough keypoints
                key_hand_points = hand_keypoints[hand_key_indices]
                
                # Convert to tensor
                hand_tensor = torch.tensor(key_hand_points, dtype=torch.float)
                x_list.append(hand_tensor)
                
                # Create hand edges
                for src, dst in hand_edges:
                    edge_index_list.append((src + node_offset, dst + node_offset))
                    edge_index_list.append((dst + node_offset, src + node_offset))  # Bidirectional
                
                # Connect body wrist to hand wrist if both are present
                if body_keypoints is not None and body_keypoints.size > 0:
                    # Add bidirectional edge between body wrist and hand wrist
                    edge_index_list.append((body_key_indices.index(body_wrist_index), node_offset))  # Body wrist to hand wrist
                    edge_index_list.append((node_offset, body_key_indices.index(body_wrist_index)))  # Hand wrist to body wrist
                
                node_offset += hand_tensor.shape[0]
        
        # Add object locations
        if object_locations is not None and (isinstance(object_locations, np.ndarray) and object_locations.size > 0):
            # Convert to tensor
            object_tensor = torch.tensor(object_locations, dtype=torch.float)
            x_list.append(object_tensor)
            
            # Connect objects to key body and hand joints
            if body_keypoints is not None and body_keypoints.size > 0:
                for i in range(object_tensor.shape[0]):
                    # Connect to body wrist
                    wrist_idx = body_key_indices.index(body_wrist_index)
                    edge_index_list.append((wrist_idx, node_offset + i))
                    edge_index_list.append((node_offset + i, wrist_idx))
            
            if hand_keypoints is not None and (isinstance(hand_keypoints, np.ndarray) and hand_keypoints.size > 0):
                for i in range(object_tensor.shape[0]):
                    # Connect to hand fingertips (indices 1-5 in our reduced representation)
                    for tip_idx in range(1, 6):  # Indices 1-5 are the fingertips in our reduced hand keypoints
                        hand_offset = len(body_key_indices) if body_keypoints is not None else 0
                        edge_index_list.append((hand_offset + tip_idx, node_offset + i))
                        edge_index_list.append((node_offset + i, hand_offset + tip_idx))
        
        # Combine node features
        if x_list:
            x = torch.cat(x_list, dim=0)
        else:
            # Default to a single node with 2D coordinates
            x = torch.zeros((1, 2), dtype=torch.float)
        
        # Create edge index tensor
        if edge_index_list:
            edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        else:
            # Default to empty edge index
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        # Create PyTorch Geometric Data object
        graph = Data(x=x, edge_index=edge_index)

        


        #########visualize part########################
        
        # # Convert to networkx graph for visualization
        # G = to_networkx(graph, to_undirected=True)
        
        # # Get node positions from x coordinates
        # # Adjust for OpenPose coordinate system (flip y-axis since 0,0 is top-left)
        # pos = {i: (float(graph.x[i][0]), -float(graph.x[i][1])) for i in range(graph.x.shape[0])}
        
        # # Create figure
        # plt.figure(figsize=(10, 8))
        
        # # Define node colors based on type
        # body_nodes = list(range(len(body_key_indices))) if body_keypoints is not None else []
        # hand_nodes = list(range(len(body_key_indices), len(body_key_indices) + len(hand_key_indices))) if hand_keypoints is not None else []
        # object_nodes = list(range(node_offset, node_offset + len(object_locations))) if object_locations is not None else []
        
        # # Draw nodes
        # nx.draw_networkx_nodes(G, pos, nodelist=body_nodes, node_color='blue', node_size=100, label='Body')
        # nx.draw_networkx_nodes(G, pos, nodelist=hand_nodes, node_color='green', node_size=80, label='Hand')
        # nx.draw_networkx_nodes(G, pos, nodelist=object_nodes, node_color='red', node_size=120, label='Object')
        
        # # Draw edges
        # nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.7)
        
        # # Add labels with proper anatomical terms
        # if impaired_side == 'right':
        #     body_labels = {i: ['Neck', 'RShoulder', 'RElbow', 'RWrist', 'MidHip'][i] for i in body_nodes}
        # else:
        #     body_labels = {i: ['Neck', 'LShoulder', 'LElbow', 'LWrist', 'MidHip'][i] for i in body_nodes}
            
        # hand_labels = {i: ['Wrist', 'Thumb', 'Index', 'Middle', 'Ring', 'Pinky'][i-len(body_key_indices)] for i in hand_nodes}
        # object_labels = {i: f'Obj{i-node_offset}' for i in object_nodes}
        
        # # Combine all labels
        # node_labels = {**body_labels, **hand_labels, **object_labels}
        
        # # Draw labels
        # nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)
        
        # # Add title with view information
        # plt.title(f'Keypoint Graph - {impaired_side.capitalize()} Side Impaired (Ipsilateral Camera View)')
        # plt.legend(loc='upper right')
        
        # # Remove axis ticks to match your example
        # plt.axis('off')
        
        # # Add perspective indicator
        # if impaired_side == 'left':
        #     plt.figtext(0.02, 0.02, "Camera viewing from patient's left side", fontsize=8)
        # else:
        #     plt.figtext(0.02, 0.02, "Camera viewing from patient's right side", fontsize=8)
        
        # plt.show()
    
        return graph


def load_data(segment_db_path, view_type, seq_length=20, batch_size=8, 
              include_hand=True, include_object=True, balance_classes=False,
              num_workers=4, test_size=0.2, val_size=0.1, random_state=42):
    """
    Load and prepare data for training
    
    Args:
        segment_db_path: Path to segment database pickle file
        view_type: Camera view type ('top' or 'ipsi')
        seq_length: Maximum sequence length
        batch_size: Batch size for data loaders
        include_hand: Whether to include hand keypoints
        include_object: Whether to include object locations
        balance_classes: Whether to balance classes in training data
        num_workers: Number of workers for data loading
        test_size: Proportion of data to use for testing
        val_size: Proportion of training data to use for validation
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary containing train, val, and test data loaders and dataset info
    """
    print(f"Loading segment database from {segment_db_path}...")
    
    # Load segment database
    with open(segment_db_path, 'rb') as f:
        segment_data = pickle.load(f)
    
    print(f"Loaded {len(segment_data)} segments")
    
    # Create dataset
    dataset = KeypointDataset(
        segment_data=segment_data,
        view_type=view_type,
        seq_length=seq_length,
        include_hand=include_hand,
        include_object=include_object
    )
    
    # Get segment IDs
    segment_ids = dataset.segment_ids
    
    # Get labels for each segment
    labels = [segment_data[sid]['label'] for sid in segment_ids]
    
    # Split into train, validation, and test sets
    train_val_ids, test_ids, train_val_labels, test_labels = train_test_split(
        segment_ids, labels, test_size=test_size, stratify=labels, random_state=random_state
    )
    
    train_ids, val_ids = train_test_split(
        train_val_ids, test_size=val_size/(1-test_size), 
        stratify=train_val_labels, random_state=random_state
    )
    
    # Balance classes in training data if requested
    if balance_classes:
        train_labels = [segment_data[sid]['label'] for sid in train_ids]
        class_counts = np.bincount(train_labels)
        max_count = np.max(class_counts)
        
        # Oversample minority classes
        balanced_train_ids = []
        for class_idx in range(len(class_counts)):
            class_ids = [sid for sid, label in zip(train_ids, train_labels) if label == class_idx]
            
            # Oversample to match the largest class
            oversampled_ids = np.random.choice(class_ids, size=max_count, replace=True).tolist()
            balanced_train_ids.extend(oversampled_ids)
        
        # Shuffle the balanced IDs
        np.random.shuffle(balanced_train_ids)
        train_ids = balanced_train_ids
    
    # Create subsetting datasets
    train_dataset = torch.utils.data.Subset(dataset, [segment_ids.index(sid) for sid in train_ids])
    val_dataset = torch.utils.data.Subset(dataset, [segment_ids.index(sid) for sid in val_ids])
    test_dataset = torch.utils.data.Subset(dataset, [segment_ids.index(sid) for sid in test_ids])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, collate_fn=collate_fn
    )
    
    # Calculate class weights for loss function
    all_labels = np.array(labels)
    class_counts = np.bincount(all_labels)
    n_samples = len(all_labels)
    n_classes = len(class_counts)
    
    class_weights = n_samples / (n_classes * class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    
    # Class information
    class_info = {
        'class_counts': class_counts,
        'class_weights': class_weights,
        'n_classes': n_classes
    }
    
    # Dataset information
    dataset_info = {
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'test_size': len(test_dataset),
        'total_size': len(dataset),
        'view_type': view_type,
        'seq_length': seq_length,
        'include_hand': include_hand,
        'include_object': include_object,
        'balanced': balance_classes
    }
    
    # Print dataset information
    print(f"Dataset split:")
    print(f"  Train: {len(train_dataset)} segments")
    print(f"  Validation: {len(val_dataset)} segments")
    print(f"  Test: {len(test_dataset)} segments")
    
    print(f"Class distribution:")
    for i, count in enumerate(class_counts):
        print(f"  Class {i}: {count} segments")
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'class_info': class_info,
        'dataset_info': dataset_info
    }


def collate_fn(batch):
    """
    Custom collate function for batching graph data
    
    Args:
        batch: List of (graphs, label, segment_id) tuples
        
    Returns:
        Tuple of (graphs, labels, segment_ids, seq_lengths)
    """
    graphs, labels, segment_ids = zip(*batch)
    
    # Get sequence lengths
    seq_lengths = [len(g) for g in graphs]
    
    # Convert labels to tensor
    labels = torch.tensor(labels, dtype=torch.long)
    
    return graphs, labels, segment_ids, seq_lengths
