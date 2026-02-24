import os
import sys
import time
from datetime import datetime
import logging
import multiprocessing
# Import our modules
from multi_pickle_processor_task import MultiPickleTaskProcessor
# from main_script import train_model, test, cross_validate
# from graph_visualization import visualize_task_graphs

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_pipeline(mode='process_and_train'):
    """
    Run the full pipeline from data processing to model training
    
    Args:
        mode: Operation mode
            - 'process_only': Only process pickle files and create task database
            - 'visualize_only': Only visualize graphs from task database
            - 'train_only': Only train model using existing task database
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
            'pickle_dir': 'D:/nature_everything/nature_dataset/task_dataset',            # Directory with task files and therapist labels
            'openpose': 'D:/pickle_files',            # Directory with body keypoint data
            'hand': 'D:/pickle_files_hand',           # Directory with hand keypoint data
            'object': 'D:/pickle_files_object'        # Directory with object location data
        },
        'csv_dir': 'D:/nature_everything',
        'ipsi_contra_csv': 'D:/nature_everything/camera_assignments.csv',   # CSV mapping patient IDs to camera IDs
        'live_rating_csv': 'D:/nature_everything/live_rating_cleaned.csv', # CSV with clinician live ratings per activity
        'output_dir': 'D:/nature_everything/combined_tasks',         # Output directory for task database
        'train_db_filename': 'train_task_database.pkl',  # Output filename for training database
        'inference_db_filename': 'inference_task_database.pkl',  # Output filename for inference database
        
        # Video and task parameters
        'fps': 30,                                    # Frames per second of the original videos
        
        # Visualization parameters
        'visualize_num_samples': 20,                  # Number of samples to visualize
        'visualize_random_seed': 1206177,                  # Random seed for visualization sampling
        
        # Camera/view configuration
        'view_type': 'ipsi',                          # Main view type to use ('top' or 'ipsi')
        
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
        config['train_db_filename'] = f"train_task_database_ipsi.pkl"
        config['inference_db_filename'] = f"inference_task_database_ipsi.pkl"
    if config['view_type'] == 'top':
        config['train_db_filename'] = f"train_task_database_top.pkl"
        config['inference_db_filename'] = f"inference_task_database_top.pkl"    
    # Create output directories
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(config['model_output_dir'], exist_ok=True)
    
    # Full paths to task databases
    train_db_path = os.path.join(config['output_dir'], config['train_db_filename'])
    inference_db_path = os.path.join(config['output_dir'], config['inference_db_filename'])
    
    # Process pickle files if needed
    if mode in ['process_only', 'process_and_train', 'process_visualize_train', 'cross_validate', 'inference']:
        logger.info("\n" + "="*80)
        logger.info("STEP 1: Processing task data and extracting keypoints")
        logger.info("="*80)
        
        processor = MultiPickleTaskProcessor(
            pickle_dirs=config['pickle_dirs'],
            output_dir=config['output_dir'],
            ipsi_contra_csv=config['ipsi_contra_csv'],
            live_rating_csv=config['live_rating_csv'],
            fps=config['fps'],
            view_type=config['view_type']
        )
        
        # Process data and build task databases
        db_paths = processor.process(view_types=[config['view_type']])
        train_db_path = db_paths['train_task_db_path']
        inference_db_path = db_paths['inference_task_db_path']
        
        process_time = time.time() - start_time
        logger.info(f"\nProcessing completed in {process_time:.2f} seconds")    
    # Print total execution time
    total_time = time.time() - start_time
    logger.info("\n" + "="*80)
    logger.info(f"Pipeline completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    logger.info("="*80)
    
    # Return paths to outputs
    return {
        'train_task_db_path': train_db_path,
        'inference_task_db_path': inference_db_path,
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