import os
import sys
import time
from datetime import datetime
import logging
import multiprocessing
# Import our modules
from multi_pickle_processor_updated import MultiPickleProcessor
from main_script import train_model, test, cross_validate
from graph_visualization import visualize_segment_graphs

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
            - 'visualize_only': Only visualize graphs from segment database
            - 'train_only': Only train model using existing segment database
            - 'process_and_train': Process pickle files and train model
            - 'process_visualize_train': Process, visualize and train
            - 'cross_validate': Process pickle files and run cross-validation
            - 'inference': Process and prepare for inference
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
        'ipsi_contra_csv': 'D:/Github/Multi_view-automatic-assessment/camera_assignments.csv',   # CSV mapping patient IDs to camera IDs
        'output_dir': 'D:/Github/GNN_Transformer/combined_segments',         # Output directory for segment database
        'train_db_filename': 'train_segment_database.pkl',  # Output filename for training database
        'inference_db_filename': 'inference_segment_database.pkl',  # Output filename for inference database
        
        # Video and segment parameters
        'fps': 30,                                    # Frames per second of the original videos
        
        # Visualization parameters
        'visualize_num_samples': 20,                  # Number of samples to visualize
        'visualize_random_seed': 1206177,                  # Random seed for visualization sampling
        
        # Camera/view configuration
        'view_type': 'top',                          # Main view type to use ('top' or 'ipsi')
        
        # Model parameters
        'model_output_dir': './output/gnn_transformer',
        'epochs': 30,
        'batch_size': 8,
        'lr': 1e-3,
        'weight_decay': 1e-5,
        'seq_length': 20,                             # Set to match num_target_frames
        'gnn_hidden': 64,
        'gnn_out': 128,
        'transformer_heads': 4,
        'transformer_layers': 4,
        'dropout': 0.2,
        'seed': 1206177,
        'balance_classes': True,
        'cross_val_folds': 5,
        'include_hand':False,
        'include_object':False,
        'num_workers': 0,  # Use all available cores
    }
    
    # Track execution time
    start_time = time.time()
    

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Keep the base directory consistent (no timestamp)
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(config['model_output_dir'], exist_ok=True)

    # Update database filenames to include timestamp
    if config['view_type'] == 'ipsi':
        config['train_db_filename'] = f"train_segment_database_ipsi.pkl"
        config['inference_db_filename'] = f"inference_segment_database_ispi.pkl"
    if config['view_type'] == 'top':
        config['train_db_filename'] = f"train_segment_database_top.pkl"
        config['inference_db_filename'] = f"inference_segment_database_top.pkl"    
    # Create output directories
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(config['model_output_dir'], exist_ok=True)
    
    # Full paths to segment databases
    train_db_path = os.path.join(config['output_dir'], config['train_db_filename'])
    inference_db_path = os.path.join(config['output_dir'], config['inference_db_filename'])
    
    # Process pickle files if needed
    if mode in ['process_only', 'process_and_train', 'process_visualize_train', 'cross_validate', 'inference']:
        logger.info("\n" + "="*80)
        logger.info("STEP 1: Processing segment data and extracting keypoints")
        logger.info("="*80)
        
        processor = MultiPickleProcessor(
            pickle_dirs=config['pickle_dirs'],
            output_dir=config['output_dir'],
            ipsi_contra_csv=config['ipsi_contra_csv'],
            fps=config['fps'],
            view_type=config['view_type']  # Add this parameter
        )
        
        # Process data and build segment databases
        db_paths = processor.process(view_types=[config['view_type']])
        train_db_path = db_paths['train_segment_db_path']
        inference_db_path = db_paths['inference_segment_db_path']
        
        process_time = time.time() - start_time
        logger.info(f"\nProcessing completed in {process_time:.2f} seconds")
    
    # Visualize graphs if needed
    if mode in ['visualize_only', 'process_visualize_train']:
        logger.info("\n" + "="*80)
        logger.info("STEP 2: Visualizing keypoint graphs")
        logger.info("="*80)
        
        vis_start_time = time.time()
        
        # Visualize training segments
        vis_dir_train = visualize_segment_graphs(
            segment_db_path=train_db_path,
            output_dir=os.path.join(config['model_output_dir'], 'train_visualizations'),
            view_type=config['view_type'],
            num_samples=config['visualize_num_samples'],
            random_seed=config['visualize_random_seed']
        )
        
        logger.info(f"Training graph visualizations saved to {vis_dir_train}")
        
        # Visualize inference segments
        if os.path.exists(inference_db_path):
            vis_dir_inference = visualize_segment_graphs(
                segment_db_path=inference_db_path,
                output_dir=os.path.join(config['model_output_dir'], 'inference_visualizations'),
                view_type=config['view_type'],
                num_samples=config['visualize_num_samples'],
                random_seed=config['visualize_random_seed']
            )
            
            logger.info(f"Inference graph visualizations saved to {vis_dir_inference}")
        
        vis_time = time.time() - vis_start_time
        logger.info(f"\nVisualization completed in {vis_time:.2f} seconds")
    
    # Train model if needed
    if mode in ['train_only', 'process_and_train', 'process_visualize_train']:
        step_num = 2 if mode != 'process_visualize_train' else 3
        logger.info("\n" + "="*80)
        logger.info(f"STEP {step_num}: Training GNN + Transformer model")
        logger.info("="*80)
        
        train_start_time = time.time()

        train_model(
            segment_db_path=train_db_path,  # Use training segment database
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
            balance_classes=config['balance_classes'],
            include_hand=config['include_hand'], 
            include_object=config['include_object'],
            num_workers=config['num_workers']  # Add this parameter
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
            segment_db_path=train_db_path,  # Use training segment database
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
    
    # Run inference if needed
    if mode == 'inference':
        logger.info("\n" + "="*80)
        logger.info("STEP 2: Running inference with pre-trained model")
        logger.info("="*80)
        
        inference_start_time = time.time()
        
        # Note: You would need to implement an inference function that can handle
        # the inference segment database with t1_label and t2_label
        # This is a placeholder for now
        logger.info("Inference functionality to be implemented")
        
        inference_time = time.time() - inference_start_time
        logger.info(f"\nInference completed in {inference_time:.2f} seconds")
    
    # Print total execution time
    total_time = time.time() - start_time
    logger.info("\n" + "="*80)
    logger.info(f"Pipeline completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    logger.info("="*80)
    
    # Return paths to outputs
    return {
        'train_segment_db_path': train_db_path,
        'inference_segment_db_path': inference_db_path,
        'model_output_dir': config['model_output_dir']
    }


if __name__ == "__main__":
    # Parse command line argument
    mode = 'process_only'  # Default mode - now includes visualization step
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        valid_modes = [
            'process_only', 
            'visualize_only', 
            'train_only', 
            'process_and_train', 
            'process_visualize_train', 
            'cross_validate', 
            'inference'
        ]
        if mode not in valid_modes:
            print(f"Invalid mode: {mode}")
            print(f"Available modes: {', '.join(valid_modes)}")
            sys.exit(1)
    
    # Run pipeline
    run_pipeline(mode)