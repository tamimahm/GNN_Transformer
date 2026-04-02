import os
import sys
import time
from datetime import datetime
import logging
import multiprocessing
# Import our modules
from multi_pickle_processor_hand_preshaping import MultiPickleHandPreshapingProcessor

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_pipeline(mode='process_and_train'):
    """
    Run the full pipeline from data processing to model training
    for hand preshaping data.

    Args:
        mode: Operation mode
            - 'process_only': Only process pickle files and create task database
            - 'train_only': Only train model using existing task database
            - 'process_and_train': Process pickle files and train model
            - 'cross_validate': Process pickle files and run cross-validation
            - 'inference': Process and prepare for inference
    """
    logger.info(f"Running hand preshaping pipeline in mode: {mode}")

    # Configuration
    config = {
        # Data paths
        'pickle_dirs': {
            'pickle_dir': 'D:/nature_everything/nature_dataset/hand_preshaping_exercise1.pkl',  # Single hand preshaping pickle file
            'openpose': 'D:/pickle_files',            # Directory with body keypoint data
            'hand': 'D:/pickle_files_hand',           # Directory with hand keypoint data
            'object': 'D:/pickle_files_object'        # Directory with object location data
        },
        'csv_dir': 'D:/nature_everything',
        'ipsi_contra_csv': 'D:/nature_everything/camera_assignments.csv',
        'live_rating_csv': 'D:/nature_everything/live_rating_cleaned.csv',
        'output_dir': 'D:/nature_everything/combined_hand_preshaping',
        'train_db_filename': 'train_hand_preshaping_database.pkl',
        'inference_db_filename': 'inference_hand_preshaping_database.pkl',

        # Video and task parameters
        'fps': 30,

        # Visualization parameters
        'visualize_num_samples': 20,
        'visualize_random_seed': 1206177,

        # Camera/view configuration
        'view_type': 'ipsi',

        # Model parameters
        'model_output_dir': './output/gnn_transformer_hand_preshaping',
        'epochs': 30,
        'batch_size': 8,
        'lr': 1e-3,
        'weight_decay': 1e-5,
        'seq_length': 20,
        'gnn_hidden': 64,
        'gnn_out': 128,
        'transformer_heads': 4,
        'transformer_layers': 4,
        'dropout': 0.2,
        'seed': 1206177,
        'balance_classes': True,
        'cross_val_folds': 5,
        'include_hand': False,
        'include_object': False,
        'num_workers': 0,
    }

    # Track execution time
    start_time = time.time()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Keep the base directory consistent (no timestamp)
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(config['model_output_dir'], exist_ok=True)

    # Update database filenames based on view type
    if config['view_type'] == 'ipsi':
        config['train_db_filename'] = f"train_hand_preshaping_database_ipsi.pkl"
        config['inference_db_filename'] = f"inference_hand_preshaping_database_ipsi.pkl"
    if config['view_type'] == 'top':
        config['train_db_filename'] = f"train_hand_preshaping_database_top.pkl"
        config['inference_db_filename'] = f"inference_hand_preshaping_database_top.pkl"

    # Create output directories
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(config['model_output_dir'], exist_ok=True)

    # Full paths to task databases
    train_db_path = os.path.join(config['output_dir'], config['train_db_filename'])
    inference_db_path = os.path.join(config['output_dir'], config['inference_db_filename'])

    # Process pickle files if needed
    if mode in ['process_only', 'process_and_train', 'cross_validate', 'inference']:
        logger.info("\n" + "="*80)
        logger.info("STEP 1: Processing hand preshaping data and extracting keypoints")
        logger.info("="*80)

        processor = MultiPickleHandPreshapingProcessor(
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
    mode = 'process_only'  # Default mode

    if len(sys.argv) > 1:
        mode = sys.argv[1]
        valid_modes = [
            'process_only',
            'train_only',
            'process_and_train',
            'cross_validate',
            'inference'
        ]
        if mode not in valid_modes:
            print(f"Invalid mode: {mode}")
            print(f"Available modes: {', '.join(valid_modes)}")
            sys.exit(1)

    # Run pipeline
    run_pipeline(mode)
