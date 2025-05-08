import os
import sys
import time
from datetime import datetime
import logging

# Import our modules
from multi_pickle_processor_updated import MultiPickleProcessor
from main_script import train_model, test, cross_validate

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_pipeline(mode='process_and_train'):
    """
    Run the full pipeline from data processing to model training
    
    Args:
        mode: Operation mode
            - 'process_only': Only process pickle files and create segment database
            - 'train_only': Only train model using existing segment database
            - 'process_and_train': Process pickle files and train model
            - 'cross_validate': Process pickle files and run cross-validation
    """
    logger.info(f"Running pipeline in mode: {mode}")
    
    # Configuration
    config = {
        # Data paths
        'pickle_dirs': {
            'pickle_dir': 'D:/pickle_dir',            # Directory with segment files and therapist labels
            'openpose': 'D:/pickle_files',            # Directory with body keypoint data
            'hand': 'D:/pickle_files_hand',           # Directory with hand keypoint data
            'object': 'D:/pickle_files_object'        # Directory with object location data
        },
        'csv_dir': 'D:/files_database',                    # Directory with segment timing CSVs
        'ipsi_contra_csv': 'D:/camera_mapping.csv',   # CSV mapping patient IDs to camera IDs
        'output_dir': 'D:/Github/Gnn_transformer/combined_segments',         # Output directory for segment database
        'segment_db_filename': 'segment_database.pkl',# Output filename for segment database
        
        # Video and segment parameters
        'fps': 30,                                    # Frames per second of the original videos
        'num_target_frames': 20,                      # Number of frames to extract per segment
        
        # Camera/view configuration
        'view_type': 'top',                           # Main view type to use ('top' or 'ipsi')
        
        # Model parameters
        'model_output_dir': './output/gnn_transformer',
        'epochs': 30,
        'batch_size': 8,
        'lr': 1e-4,
        'weight_decay': 1e-5,
        'seq_length': 20,                             # Set to match num_target_frames
        'gnn_hidden': 64,
        'gnn_out': 128,
        'transformer_heads': 4,
        'transformer_layers': 4,
        'dropout': 0.2,
        'seed': 42,
        'balance_classes': True,
        'cross_val_folds': 5
    }
    
    # Track execution time
    start_time = time.time()
    
    # Create timestamp for output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Update output directories with timestamp
    config['output_dir'] = f"{config['output_dir']}_{timestamp}"
    config['model_output_dir'] = f"{config['model_output_dir']}_{timestamp}"
    
    # Create output directories
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(config['model_output_dir'], exist_ok=True)
    
    # Full path to segment database
    segment_db_path = os.path.join(config['output_dir'], config['segment_db_filename'])
    
    # Process pickle files if needed
    if mode in ['process_only', 'process_and_train', 'cross_validate']:
        logger.info("\n" + "="*80)
        logger.info("STEP 1: Processing segment data and extracting keypoints")
        logger.info("="*80)
        
        processor = MultiPickleProcessor(
            pickle_dirs=config['pickle_dirs'],
            csv_dir=config['csv_dir'],
            output_dir=config['output_dir'],
            ipsi_contra_csv=config['ipsi_contra_csv'],
            fps=config['fps']
        )
        
        # Process all data and build segment database
        segment_db_path = processor.process(
            view_types=[config['view_type']],
            output_filename=config['segment_db_filename'],
            num_target_frames=config['num_target_frames']
        )
        
        process_time = time.time() - start_time
        logger.info(f"\nProcessing completed in {process_time:.2f} seconds")
    
    # Train model if needed
    if mode in ['train_only', 'process_and_train']:
        logger.info("\n" + "="*80)
        logger.info("STEP 2: Training GNN + Transformer model")
        logger.info("="*80)
        
        train_start_time = time.time()
        
        train_model(
            segment_db_path=segment_db_path,
            output_dir=config['model_output_dir'],
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
            balance_classes=config['balance_classes']
        )
        
        train_time = time.time() - train_start_time
        logger.info(f"\nTraining completed in {train_time:.2f} seconds")
    
    # Run cross-validation if needed
    if mode == 'cross_validate':
        logger.info("\n" + "="*80)
        logger.info(f"STEP 2: Running {config['cross_val_folds']}-fold cross-validation")
        logger.info("="*80)
        
        cv_start_time = time.time()
        
        cross_validate(
            segment_db_path=segment_db_path,
            view_type=config['view_type'],
            output_dir=config['model_output_dir'],
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
            balance_classes=config['balance_classes']
        )
        
        cv_time = time.time() - cv_start_time
        logger.info(f"\nCross-validation completed in {cv_time:.2f} seconds")
    
    # Print total execution time
    total_time = time.time() - start_time
    logger.info("\n" + "="*80)
    logger.info(f"Pipeline completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    logger.info("="*80)
    
    # Return paths to outputs
    return {
        'segment_db_path': segment_db_path,
        'model_output_dir': config['model_output_dir']
    }


if __name__ == "__main__":
    # Parse command line argument
    mode = 'process_and_train'  # Default mode
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode not in ['process_only', 'train_only', 'process_and_train', 'cross_validate']:
            print(f"Invalid mode: {mode}")
            print("Available modes: process_only, train_only, process_and_train, cross_validate")
            sys.exit(1)
    
    # Run pipeline
    run_pipeline(mode)