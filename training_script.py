import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import argparse
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

# Import our modules
from data_loader import load_data
from gnn_transformer_updated import GNNTransformerModel


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train for one epoch
    
    Args:
        model: Model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use
        
    Returns:
        Average training loss and accuracy
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # Progress bar
    pbar = tqdm(train_loader, desc="Training")
    
    for graphs, labels, segment_ids, seq_lengths in pbar:
        # Move data to device
        labels = labels.to(device)
        graphs = [[g.to(device) for g in seq] for seq in graphs]
        
        # Forward pass
        outputs = model(graphs, seq_lengths)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': total_loss / (pbar.n + 1),
            'acc': 100 * correct / total
        })
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """
    Validate the model
    
    Args:
        model: Model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to use
        
    Returns:
        Validation loss, accuracy, and predictions
    """
    model.eval()
    total_loss = 0
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for graphs, labels, segment_ids, seq_lengths in val_loader:
            # Move data to device
            labels = labels.to(device)
            graphs = [[g.to(device) for g in seq] for seq in graphs]
            
            # Forward pass
            outputs = model(graphs, seq_lengths)
            loss = criterion(outputs, labels)
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            
            # Store for metrics
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    # Calculate metrics
    accuracy = 100 * accuracy_score(all_labels, all_predictions)
    f1 = 100 * f1_score(all_labels, all_predictions, average='weighted')
    avg_loss = total_loss / len(val_loader)
    
    return avg_loss, accuracy, f1, all_labels, all_predictions


def test(model, test_loader, criterion, device, output_dir):
    """
    Test the model and generate evaluation metrics
    
    Args:
        model: Model to test
        test_loader: Test data loader
        criterion: Loss function
        device: Device to use
        output_dir: Output directory for results
        
    Returns:
        Test accuracy, F1 score, and detailed metrics
    """
    # Run validation function on test set
    test_loss, accuracy, f1, true_labels, predictions = validate(
        model, test_loader, criterion, device
    )
    
    # Generate confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    
    # Generate classification report
    report = classification_report(true_labels, predictions, output_dict=True)
    
    # Save results
    results = {
        'test_loss': test_loss,
        'accuracy': accuracy,
        'f1_score': f1,
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as JSON
    with open(os.path.join(output_dir, 'test_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    
    print(f"Test results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  F1 Score: {f1:.2f}%")
    print(f"  Results saved to {output_dir}")
    
    return accuracy, f1, results


def visualize_attention(model, data_loader, device, output_dir, num_samples=5):
    """
    Visualize attention weights from transformer layers
    
    Args:
        model: Trained model
        data_loader: Data loader
        device: Device to use
        output_dir: Output directory for visualizations
        num_samples: Number of samples to visualize
    """
    model.eval()
    
    # Create output directory
    vis_dir = os.path.join(output_dir, 'attention_visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Get samples
    samples = []
    with torch.no_grad():
        for graphs, labels, segment_ids, seq_lengths in data_loader:
            for i in range(min(len(graphs), num_samples - len(samples))):
                sample_graphs = [graphs[i]]
                sample_label = labels[i].item()
                sample_seq_length = [seq_lengths[i]]
                sample_id = segment_ids[i]
                
                # Move to device
                sample_graphs = [[g.to(device) for g in seq] for seq in sample_graphs]
                
                # Get attention weights
                attention_weights = model.get_attention_weights(sample_graphs, sample_seq_length)
                
                samples.append({
                    'graphs': sample_graphs,
                    'label': sample_label,
                    'segment_id': sample_id,
                    'seq_length': sample_seq_length[0],
                    'attention_weights': attention_weights
                })
            
            if len(samples) >= num_samples:
                break
    
    # Visualize attention weights for each sample
    for i, sample in enumerate(samples):
        attention_weights = sample['attention_weights']
        seq_length = sample['seq_length']
        
        # Create figure with subplots for each layer
        num_layers = len(attention_weights)
        fig, axes = plt.subplots(1, num_layers, figsize=(6 * num_layers, 5))
        
        # Handle case of single layer
        if num_layers == 1:
            axes = [axes]
        
        # Plot attention weights for each layer
        for j, attn in enumerate(attention_weights):
            # Get attention matrix for this sample (first in batch)
            attn_matrix = attn[0, :seq_length, :seq_length].cpu().numpy()
            
            # Plot heatmap
            sns.heatmap(attn_matrix, ax=axes[j], cmap='viridis', vmin=0, vmax=1)
            axes[j].set_title(f"Layer {j+1}")
            axes[j].set_xlabel("Frame (Key)")
            axes[j].set_ylabel("Frame (Query)")
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f"attention_sample_{i}_label_{sample['label']}.png"))
        plt.close()
    
    print(f"Attention visualizations saved to {vis_dir}")


def visualize_keypoints(data_loader, output_dir, num_samples=5):
    """
    Visualize keypoints from samples
    
    Args:
        data_loader: Data loader
        output_dir: Output directory for visualizations
        num_samples: Number of samples to visualize
    """
    # Create output directory
    vis_dir = os.path.join(output_dir, 'keypoint_visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Get samples
    samples = []
    for graphs, labels, segment_ids, seq_lengths in data_loader:
        for i in range(min(len(graphs), num_samples - len(samples))):
            sample_graphs = graphs[i]
            sample_label = labels[i].item()
            sample_seq_length = seq_lengths[i]
            sample_id = segment_ids[i]
            
            samples.append({
                'graphs': sample_graphs,
                'label': sample_label,
                'segment_id': sample_id,
                'seq_length': sample_seq_length
            })
        
        if len(samples) >= num_samples:
            break
    
    # Visualize keypoints for each sample
    for i, sample in enumerate(samples):
        graph_sequence = sample['graphs']
        seq_length = sample['seq_length']
        
        # Create figure with subplots for selected frames
        # Show first frame, middle frame, and last frame
        frame_indices = [0, seq_length//2, seq_length-1]
        fig, axes = plt.subplots(1, len(frame_indices), figsize=(15, 5))
        
        for j, frame_idx in enumerate(frame_indices):
            if frame_idx < len(graph_sequence):
                graph = graph_sequence[frame_idx]
                # Get node positions
                pos = graph.x.cpu().numpy()
                
                # Get edge connections
                edge_index = graph.edge_index.cpu().numpy()
                
                # Plot nodes
                axes[j].scatter(pos[:, 0], pos[:, 1], c='blue', s=20)
                
                # Plot edges
                for e in range(edge_index.shape[1]):
                    src, dst = edge_index[0, e], edge_index[1, e]
                    if src < pos.shape[0] and dst < pos.shape[0]:  # Ensure valid indices
                        axes[j].plot([pos[src, 0], pos[dst, 0]], 
                                    [pos[src, 1], pos[dst, 1]], 'k-', alpha=0.3)
                
                axes[j].set_title(f"Frame {frame_idx}")
                axes[j].set_aspect('equal')
                axes[j].invert_yaxis()  # Invert y-axis to match image coordinates
            
            else:
                axes[j].set_visible(False)
        
        plt.suptitle(f"Sample {i} (Label: {sample['label']})")
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f"keypoints_sample_{i}_label_{sample['label']}.png"))
        plt.close()
    
    print(f"Keypoint visualizations saved to {vis_dir}")


def cross_validate(segment_db_path, view_type, output_dir, num_folds=5, **kwargs):
    """
    Perform cross-validation
    
    Args:
        segment_db_path: Path to segment database
        view_type: Camera view type ('top' or 'ipsi')
        output_dir: Output directory
        num_folds: Number of cross-validation folds
        **kwargs: Additional arguments for training
        
    Returns:
        Average metrics across folds
    """
    # Load segment database
    print(f"Loading segment database from {segment_db_path}...")
    with open(segment_db_path, 'rb') as f:
        segment_data = pickle.load(f)
    
    # Get segment IDs for the selected view type
    valid_segment_ids = [
        sid for sid, segment in segment_data.items()
        if segment['view_type'] == view_type and segment['label'] is not None
    ]
    
    # Get labels for each segment
    labels = [segment_data[sid]['label'] for sid in valid_segment_ids]
    
    # Create folds with stratification
    fold_indices = []
    unique_labels = np.unique(labels)
    
    # Split each class into folds
    for label in unique_labels:
        label_indices = [i for i, l in enumerate(labels) if l == label]
        np.random.shuffle(label_indices)
        fold_size = len(label_indices) // num_folds
        
        for fold in range(num_folds):
            start_idx = fold * fold_size
            end_idx = (fold + 1) * fold_size if fold < num_folds - 1 else len(label_indices)
            fold_indices.append((fold, label_indices[start_idx:end_idx]))
    
    # Group indices by fold
    fold_data = []
    for fold in range(num_folds):
        fold_indices_flat = [idx for f, indices in fold_indices if f == fold for idx in indices]
        fold_data.append([valid_segment_ids[idx] for idx in fold_indices_flat])
    
    # Create output directory for cross-validation
    cv_output_dir = os.path.join(output_dir, f"cross_validation_{num_folds}fold")
    os.makedirs(cv_output_dir, exist_ok=True)
    
    # Metrics for each fold
    fold_metrics = []
    
    # Train and evaluate on each fold
    for fold in range(num_folds):
        print(f"\n{'-'*80}")
        print(f"Fold {fold+1}/{num_folds}")
        print(f"{'-'*80}")
        
        # Create train and test sets
        test_ids = fold_data[fold]
        train_ids = [sid for f in range(num_folds) if f != fold for sid in fold_data[f]]
        
        # Create temporary segment database for this fold
        fold_segment_data = {
            sid: segment_data[sid] for sid in train_ids + test_ids
        }
        
        # Create fold output directory
        fold_dir = os.path.join(cv_output_dir, f"fold_{fold+1}")
        os.makedirs(fold_dir, exist_ok=True)
        
        # Save temporary segment database
        temp_db_path = os.path.join(fold_dir, "temp_segment_db.pkl")
        with open(temp_db_path, 'wb') as f:
            pickle.dump(fold_segment_data, f)
        
        # Train on this fold
        train_kwargs = kwargs.copy()
        train_kwargs.update({
            'segment_db_path': temp_db_path,
            'output_dir': fold_dir,
            'view_type': view_type,
            'test_segment_ids': test_ids,
            'is_cross_val_fold': True
        })
        
        # Run training
        accuracy, f1, _ = train_model(**train_kwargs)
        
        # Store metrics
        fold_metrics.append({
            'fold': fold + 1,
            'accuracy': accuracy,
            'f1_score': f1
        })
    
    # Calculate average metrics
    avg_accuracy = np.mean([m['accuracy'] for m in fold_metrics])
    avg_f1 = np.mean([m['f1_score'] for m in fold_metrics])
    
    # Save cross-validation results
    cv_results = {
        'num_folds': num_folds,
        'fold_metrics': fold_metrics,
        'average_accuracy': avg_accuracy,
        'average_f1_score': avg_f1
    }
    
    with open(os.path.join(cv_output_dir, 'cross_validation_results.json'), 'w') as f:
        json.dump(cv_results, f, indent=2)
    
    print(f"\nCross-validation results:")
    print(f"  Average Accuracy: {avg_accuracy:.2f}%")
    print(f"  Average F1 Score: {avg_f1:.2f}%")
    print(f"  Results saved to {cv_output_dir}")
    
    return avg_accuracy, avg_f1, cv_results


def train_model(segment_db_path, output_dir, view_type='top', epochs=30, batch_size=8, 
                lr=1e-4, weight_decay=1e-5, seq_length=32, gnn_hidden=64, gnn_out=128,
                transformer_heads=4, transformer_layers=4, dropout=0.2, seed=42,
                balance_classes=False, num_workers=4, include_hand=True, include_object=True,
                checkpoint=None, test_segment_ids=None, is_cross_val_fold=False):
    """
    Train the GNN + Transformer model
    
    Args:
        segment_db_path: Path to segment database
        output_dir: Output directory
        view_type: Camera view type ('top' or 'ipsi')
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        weight_decay: Weight decay coefficient
        seq_length: Maximum sequence length
        gnn_hidden: GNN hidden dimension
        gnn_out: GNN output dimension
        transformer_heads: Number of transformer attention heads
        transformer_layers: Number of transformer layers
        dropout: Dropout rate
        seed: Random seed
        balance_classes: Whether to balance classes in training
        num_workers: Number of data loader workers
        include_hand: Whether to include hand keypoints
        include_object: Whether to include object locations
        checkpoint: Path to model checkpoint for resuming training
        test_segment_ids: List of segment IDs for testing (for cross-validation)
        is_cross_val_fold: Whether this is a cross-validation fold
        
    Returns:
        Test accuracy, F1 score, and detailed metrics
    """
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    config = {
        'segment_db_path': segment_db_path,
        'output_dir': output_dir,
        'view_type': view_type,
        'epochs': epochs,
        'batch_size': batch_size,
        'lr': lr,
        'weight_decay': weight_decay,
        'seq_length': seq_length,
        'gnn_hidden': gnn_hidden,
        'gnn_out': gnn_out,
        'transformer_heads': transformer_heads,
        'transformer_layers': transformer_layers,
        'dropout': dropout,
        'seed': seed,
        'balance_classes': balance_classes,
        'num_workers': num_workers,
        'include_hand': include_hand,
        'include_object': include_object,
        'checkpoint': checkpoint,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    data = load_data(
        segment_db_path=segment_db_path,
        view_type=view_type,
        seq_length=seq_length,
        batch_size=batch_size,
        include_hand=include_hand,
        include_object=include_object,
        balance_classes=balance_classes,
        num_workers=num_workers
    )
    
    train_loader = data['train_loader']
    val_loader = data['val_loader']
    test_loader = data['test_loader']
    class_info = data['class_info']
    
    # Create model
    model = GNNTransformerModel(
        node_feature_dim=2,  # x, y coordinates
        gnn_hidden_dim=gnn_hidden,
        gnn_output_dim=gnn_out,
        transformer_heads=transformer_heads,
        transformer_layers=transformer_layers,
        num_classes=class_info['n_classes'],
        dropout=dropout
    )
    
    # Move model to device
    model = model.to(device)
    
    # Load checkpoint if provided
    start_epoch = 0
    if checkpoint and os.path.isfile(checkpoint):
        print(f"Loading checkpoint from {checkpoint}")
        checkpoint_data = torch.load(checkpoint, map_location=device)
        model.load_state_dict(checkpoint_data['model_state_dict'])
        start_epoch = checkpoint_data.get('epoch', 0)
        print(f"Resuming from epoch {start_epoch}")
    
    # Define loss function with class weights
    class_weights = class_info['class_weights'].to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    best_val_acc = 0
    best_epoch = 0
    
    # Lists to store metrics
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    val_f1s = []
    learning_rates = []
    
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc, val_f1, _, _ = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)
        
        # Update learning rate
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        # Print metrics
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.2f}%")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Check if this is the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1
            }, os.path.join(output_dir, 'gnn_transformer_best.pt'))
            
            print(f"  New best model saved! Val Acc: {val_acc:.2f}%")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1
            }, os.path.join(output_dir, f'gnn_transformer_epoch_{epoch+1}.pt'))
    
    # Plot training curves
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    plt.subplot(2, 2, 2)
    plt.plot(train_accs, label='Train')
    plt.plot(val_accs, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy Curves')
    
    plt.subplot(2, 2, 3)
    plt.plot(val_f1s)
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score (%)')
    plt.title('Validation F1 Score')
    
    plt.subplot(2, 2, 4)
    plt.plot(learning_rates)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    
    # Load best model for testing
    best_model_path = os.path.join(output_dir, 'gnn_transformer_best.pt')
    checkpoint_data = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint_data['model_state_dict'])
    
    print(f"\nLoaded best model from epoch {best_epoch+1}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Test the model
    test_accuracy, test_f1, test_results = test(model, test_loader, criterion, device, output_dir)
    
    # Visualize attention weights
    visualize_attention(model, test_loader, device, output_dir)
    
    # Visualize keypoints
    visualize_keypoints(test_loader, output_dir)
    
    return test_accuracy, test_f1, test_results


def main():
    """
    Main function - run GNN Transformer without using argparse
    """
    # Configuration parameters (modify as needed)
    config = {
        'mode': 'train',                           # 'train', 'test', 'cross_validate', or 'visualize'
        'segment_db_path': 'D:/combined_segments/segment_database.pkl',
        'view_type': 'top',                        # 'top' or 'ipsilateral'
        'output_dir': './output',
        
        # Model parameters
        'epochs': 30,
        'batch_size': 8,
        'lr': 1e-4,
        'weight_decay': 1e-5,
        'seq_length': 32,
        'gnn_hidden': 64,
        'gnn_out': 128,
        'transformer_heads': 4,
        'transformer_layers': 4,
        'dropout': 0.2,
        
        # Other parameters
        'num_workers': 4,
        'seed': 42,
        'checkpoint': None,                        # Path to model checkpoint
        'balance_classes': True,
        'cross_val_folds': 5,
        'include_hand': True,
        'include_object': True,
        
        # Visualization options
        'visualize_keypoints': True,
        'visualize_attention': True,
        'visualize_gnn': False,
    }
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Run selected mode
    if config['mode'] == 'train':
        print(f"Running training mode with {config['view_type']} view")
        train_model(
            segment_db_path=config['segment_db_path'],
            output_dir=config['output_dir'],
            view_type=config['view_type'],
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            lr=config['lr'],
            weight_decay=config['weight_decay'],
            seq_length=config['seq_length'],
            gnn_hidden=config['gnn_hidden'],
            gnn_out=config['gnn_out'],
            transformer_heads=config['transformer_heads'],
            transformer_layers=config['transformer_layers'],
            dropout=config['dropout'],
            seed=config['seed'],
            balance_classes=config['balance_classes'],
            num_workers=config['num_workers'],
            include_hand=config['include_hand'],
            include_object=config['include_object'],
            checkpoint=config['checkpoint']
        )
    
    elif config['mode'] == 'test':
        print(f"Running test mode with {config['view_type']} view")
        
        # Load data
        data = load_data(
            segment_db_path=config['segment_db_path'],
            view_type=config['view_type'],
            seq_length=config['seq_length'],
            batch_size=config['batch_size'],
            include_hand=config['include_hand'],
            include_object=config['include_object'],
            balance_classes=False,
            num_workers=config['num_workers']
        )
        
        test_loader = data['test_loader']
        class_info = data['class_info']
        
        # Create model
        model = GNNTransformerModel(
            node_feature_dim=2,
            gnn_hidden_dim=config['gnn_hidden'],
            gnn_output_dim=config['gnn_out'],
            transformer_heads=config['transformer_heads'],
            transformer_layers=config['transformer_layers'],
            num_classes=class_info['n_classes'],
            dropout=config['dropout']
        )
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Load checkpoint
        if config['checkpoint']:
            checkpoint_data = torch.load(config['checkpoint'], map_location=device)
            model.load_state_dict(checkpoint_data['model_state_dict'])
            print(f"Loaded model from {config['checkpoint']}")
        else:
            print("Error: No checkpoint provided for testing")
            return
        
        # Define loss function
        class_weights = class_info['class_weights'].to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Test the model
        test(model, test_loader, criterion, device, config['output_dir'])
        
        # Visualize if requested
        if config['visualize_attention']:
            visualize_attention(model, test_loader, device, config['output_dir'])
        
        if config['visualize_keypoints']:
            visualize_keypoints(test_loader, config['output_dir'])
    
    elif config['mode'] == 'cross_validate':
        print(f"Running cross-validation with {config['cross_val_folds']} folds on {config['view_type']} view")
        cross_validate(
            segment_db_path=config['segment_db_path'],
            view_type=config['view_type'],
            output_dir=config['output_dir'],
            num_folds=config['cross_val_folds'],
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            lr=config['lr'],
            weight_decay=config['weight_decay'],
            seq_length=config['seq_length'],
            gnn_hidden=config['gnn_hidden'],
            gnn_out=config['gnn_out'],
            transformer_heads=config['transformer_heads'],
            transformer_layers=config['transformer_layers'],
            dropout=config['dropout'],
            seed=config['seed'],
            balance_classes=config['balance_classes'],
            num_workers=config['num_workers'],
            include_hand=config['include_hand'],
            include_object=config['include_object']
        )
    
    elif config['mode'] == 'visualize':
        print(f"Running visualization mode with {config['view_type']} view")
        
        # Load data
        data = load_data(
            segment_db_path=config['segment_db_path'],
            view_type=config['view_type'],
            seq_length=config['seq_length'],
            batch_size=config['batch_size'],
            include_hand=config['include_hand'],
            include_object=config['include_object'],
            balance_classes=False,
            num_workers=config['num_workers']
        )
        
        test_loader = data['test_loader']
        class_info = data['class_info']
        
        # Create model
        model = GNNTransformerModel(
            node_feature_dim=2,
            gnn_hidden_dim=config['gnn_hidden'],
            gnn_output_dim=config['gnn_out'],
            transformer_heads=config['transformer_heads'],
            transformer_layers=config['transformer_layers'],
            num_classes=class_info['n_classes'],
            dropout=config['dropout']
        )
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Load checkpoint
        if config['checkpoint']:
            checkpoint_data = torch.load(config['checkpoint'], map_location=device)
            model.load_state_dict(checkpoint_data['model_state_dict'])
            print(f"Loaded model from {config['checkpoint']}")
        else:
            print("Error: No checkpoint provided for visualization")
            return
        
        # Visualize keypoints, attention, and GNN activations
        if config['visualize_keypoints']:
            visualize_keypoints(test_loader, config['output_dir'])
        
        if config['visualize_attention']:
            visualize_attention(model, test_loader, device, config['output_dir'])
        
        # Note: GNN visualization would need to be implemented
    
    else:
        print(f"Invalid mode: {config['mode']}")


if __name__ == "__main__":
    main()