import os
import pickle
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
from segment_reader import read_segment_information
from segment_utils import load_video_segments_info, extract_segment_frames, normalize_keypoints, build_segment_database

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultiPickleProcessor:
    """
    Class to process multiple pickle directories and create a combined segment database
    for the GNN + Transformer model
    """
    
    def __init__(self, 
                pickle_dirs={
                    'pickle_dir': 'D:/pickle_dir',          # Segment files with therapist labels
                    'openpose': 'D:/pickle_files',          # OpenPose body keypoint data
                    'hand': 'D:/pickle_files_hand',         # Hand keypoints from MediaPipe
                    'object': 'D:/pickle_files_object'      # Object locations from TridentNet
                },
                csv_dir='D:/csv_files',                     # Directory with segment timing CSVs
                output_dir='D:/combined_segments',
                ipsi_contra_csv='D:/camera_mapping.csv',
                num_files_per_dir=10,
                fps=30):
        """
        Initialize the processor with pickle directories
        
        Args:
            pickle_dirs: Dictionary of pickle directories (key is type, value is directory path)
            csv_dir: Directory with segment timing CSV files
            output_dir: Directory to save the combined segment database
            ipsi_contra_csv: CSV file mapping patient IDs to ipsilateral camera IDs
            num_files_per_dir: Number of pickle files to read from each directory
            fps: Frames per second of the original videos
        """
        self.pickle_dirs = pickle_dirs
        self.csv_dir = csv_dir
        self.output_dir = output_dir
        self.ipsi_contra_csv = ipsi_contra_csv
        self.num_files_per_dir = num_files_per_dir
        self.fps = fps
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize data containers
        self.body_data = {}      # Patient ID -> activity ID -> keypoints
        self.hand_data = {}      # Patient ID -> activity ID -> keypoints
        self.object_data = {}    # Patient ID -> activity ID -> object locations
        self.segment_data = {}   # Segment ID -> combined data
        self.segment_records = None  # Segment timing records from CSV
    
    def load_segment_timing_info(self):
        """
        Load segment timing information from CSV files
        """
        logger.info("Loading segment timing information...")
        self.segment_records = load_video_segments_info(self.csv_dir)
        logger.info(f"Loaded timing info for {len(self.segment_records)} video segments")
        return self.segment_records
    
    def load_pickle_files(self):
        """
        Load pickle files from each directory
        """
        logger.info("Loading pickle files...")
        
        # Load body keypoints (from pickle_files)
        if 'openpose' in self.pickle_dirs:
            self._load_openpose_pickle_files()
            
        # Load hand keypoints (from pickle_files_hand)
        if 'hand' in self.pickle_dirs:
            self._load_hand_pickle_files()
            
        # Load object locations (from pickle_files_object)
        if 'object' in self.pickle_dirs:
            self._load_object_pickle_files()
    
    def _load_openpose_pickle_files(self):
        """
        Load OpenPose pickle files (from pickle_files)
        """
        openpose_dir = self.pickle_dirs.get('openpose')
        if not openpose_dir or not os.path.exists(openpose_dir):
            logger.warning(f"OpenPose directory not found: {openpose_dir}")
            return
            
        # Find all pickle files
        pickle_files = glob.glob(os.path.join(openpose_dir, "*.pkl"))
        if len(pickle_files) == 0:
            logger.warning(f"No pickle files found in {openpose_dir}")
            return
            
        # Limit to specified number of files
        pickle_files = pickle_files[:self.num_files_per_dir]
        
        logger.info(f"Loading {len(pickle_files)} OpenPose files...")
        for pkl_file in tqdm(pickle_files, desc="Loading OpenPose files"):
            try:
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                
                # Process based on expected format
                for patient_id, patient_data in data.items():
                    if patient_id not in self.body_data:
                        self.body_data[patient_id] = {}
                    
                    # Format from Updated JSON to Pickle Processor
                    if 'top' in patient_data and 'ipsi' in patient_data:
                        for view_type in ['top', 'ipsi']:
                            for activity_id, keypoints in patient_data[view_type].items():
                                if activity_id not in self.body_data[patient_id]:
                                    self.body_data[patient_id][activity_id] = {}
                                
                                if view_type not in self.body_data[patient_id][activity_id]:
                                    self.body_data[patient_id][activity_id][view_type] = {}
                                
                                self.body_data[patient_id][activity_id][view_type] = keypoints
            except Exception as e:
                logger.error(f"Error loading {pkl_file}: {e}")
    
    def _load_hand_pickle_files(self):
        """
        Load hand keypoint pickle files (from pickle_files_hand)
        """
        hand_dir = self.pickle_dirs.get('hand')
        if not hand_dir or not os.path.exists(hand_dir):
            logger.warning(f"Hand keypoints directory not found: {hand_dir}")
            return
            
        # Find all pickle files
        pickle_files = glob.glob(os.path.join(hand_dir, "*.pkl"))
        if len(pickle_files) == 0:
            logger.warning(f"No pickle files found in {hand_dir}")
            return
            
        # Limit to specified number of files
        pickle_files = pickle_files[:self.num_files_per_dir]
        
        logger.info(f"Loading {len(pickle_files)} hand keypoint files...")
        for pkl_file in tqdm(pickle_files, desc="Loading hand keypoint files"):
            try:
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                
                # Process based on expected format
                for patient_id, patient_data in data.items():
                    if patient_id not in self.hand_data:
                        self.hand_data[patient_id] = {}
                    
                    # Assume similar format to body data
                    if 'top' in patient_data and 'ipsi' in patient_data:
                        for view_type in ['top', 'ipsi']:
                            for activity_id, keypoints in patient_data[view_type].items():
                                if activity_id not in self.hand_data[patient_id]:
                                    self.hand_data[patient_id][activity_id] = {}
                                
                                if view_type not in self.hand_data[patient_id][activity_id]:
                                    self.hand_data[patient_id][activity_id][view_type] = {}
                                
                                self.hand_data[patient_id][activity_id][view_type] = keypoints
            except Exception as e:
                logger.error(f"Error loading {pkl_file}: {e}")
    
    def _load_object_pickle_files(self):
        """
        Load object location pickle files (from pickle_files_object)
        """
        object_dir = self.pickle_dirs.get('object')
        if not object_dir or not os.path.exists(object_dir):
            logger.warning(f"Object locations directory not found: {object_dir}")
            return
            
        # Find all pickle files
        pickle_files = glob.glob(os.path.join(object_dir, "*.pkl"))
        if len(pickle_files) == 0:
            logger.warning(f"No pickle files found in {object_dir}")
            return
            
        # Limit to specified number of files
        pickle_files = pickle_files[:self.num_files_per_dir]
        
        logger.info(f"Loading {len(pickle_files)} object location files...")
        for pkl_file in tqdm(pickle_files, desc="Loading object location files"):
            try:
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                
                # Process based on expected format
                for patient_id, patient_data in data.items():
                    if patient_id not in self.object_data:
                        self.object_data[patient_id] = {}
                    
                    # Assume similar format to other data
                    if isinstance(patient_data, dict):
                        for activity_id, object_info in patient_data.items():
                            self.object_data[patient_id][activity_id] = object_info
            except Exception as e:
                logger.error(f"Error loading {pkl_file}: {e}")
    
    def load_therapist_ratings(self):
        """
        Load therapist ratings from pickle_dir
        
        Returns:
            Dictionary mapping (patient_id, activity_id, segment_id) to label
        """
        pickle_dir = self.pickle_dirs.get('pickle_dir')
        if not pickle_dir or not os.path.exists(pickle_dir):
            logger.error(f"Segment directory not found: {pickle_dir}")
            return {}
        
        # Read segment information with therapist labels
        all_segments = read_segment_information(
            pickle_dir=pickle_dir,
            ipsi_contra_csv=self.ipsi_contra_csv
        )
        
        # Create mapping from (patient_id, activity_id, segment_id) to label
        ratings = {}
        for segment in all_segments:
            key = (segment['patient_id'], segment['activity_id'], segment['segment_id'])
            ratings[key] = segment['label']
        
        logger.info(f"Loaded therapist ratings for {len(ratings)} segments")
        return ratings
    
    def build_segment_database(self, view_types=['top', 'ipsi'], num_target_frames=20):
        """
        Build segment database by combining segment timing, keypoint data, and therapist ratings
        
        Args:
            view_types: List of view types to include ('top', 'ipsi', or both)
            num_target_frames: Number of frames to extract per segment
            
        Returns:
            Dictionary with segments indexed by segment ID
        """
        logger.info("Building segment database...")
        
        # Make sure segment records are loaded
        if self.segment_records is None:
            self.load_segment_timing_info()
        
        # Load therapist ratings
        therapist_ratings = self.load_therapist_ratings()
        
        # Process body keypoints first
        body_segments = build_segment_database(
            self.segment_records, 
            self.body_data, 
            data_type='body',
            fps=self.fps, 
            num_target_frames=num_target_frames
            
        )
        
        # Process hand keypoints
        hand_segments = {}
        if self.hand_data:
            hand_segments = build_segment_database(
                self.segment_records, 
                self.hand_data, 
                data_type='hand',
                fps=self.fps, 
                num_target_frames=num_target_frames
                
            )
        
        # Process object locations
        object_segments = {}
        if self.object_data:
            object_segments = build_segment_database(
                self.segment_records, 
                self.object_data, 
                data_type='object',
                fps=self.fps, 
                num_target_frames=num_target_frames
                
            )
        
        # Combine all data into final segments
        segment_id = 0
        for key, segment in body_segments.items():
            patient_id = segment['patient_id']
            activity_id = segment['activity_id']
            seg_id = segment['segment_id']
            view_type = segment['view_type']
            
            # Skip if view type not requested
            if view_types and view_type not in view_types:
                continue
            
            # Get therapist rating
            rating_key = (patient_id, activity_id, seg_id)
            label = therapist_ratings.get(rating_key)
            
            # Skip if no rating
            if label is None:
                continue
            
            # Get hand keypoints if available
            hand_keypoints = None
            if hand_segments and key in hand_segments:
                hand_keypoints = hand_segments[key]['keypoints']
            
            # Get object locations if available
            object_locations = None
            if object_segments and key in object_segments:
                object_locations = object_segments[key]['keypoints']
            
            # Create combined segment
            self.segment_data[segment_id] = {
                'patient_id': patient_id,
                'activity_id': activity_id,
                'segment_id': seg_id,
                'camera_id': segment['camera_id'],
                'view_type': view_type,
                'start_time': segment['start_time'],
                'end_time': segment['end_time'],
                'body_keypoints': segment['keypoints'],
                'hand_keypoints': hand_keypoints,
                'object_locations': object_locations,
                'label': label
            }
            
            segment_id += 1
        
        logger.info(f"Built {len(self.segment_data)} segments with keypoint data and therapist ratings")
        
        return self.segment_data
    
    def save_segment_database(self, filename='segment_database.pkl'):
        """
        Save the segment database
        """
        output_path = os.path.join(self.output_dir, filename)
        logger.info(f"Saving segment database to {output_path}...")
        
        with open(output_path, 'wb') as f:
            pickle.dump(self.segment_data, f)
        
        logger.info(f"Saved {len(self.segment_data)} segments")
    
    def process(self, view_types=['top', 'ipsi'], output_filename='segment_database.pkl', num_target_frames=20):
        """
        Run the full processing pipeline
        
        Args:
            view_types: List of view types to include ('top', 'ipsi', or both)
            output_filename: Filename for the output segment database
            num_target_frames: Number of frames to extract per segment
            
        Returns:
            Path to saved segment database
        """
        self.load_segment_timing_info()
        self.load_pickle_files()
        self.build_segment_database(view_types, num_target_frames)
        self.save_segment_database(output_filename)
        
        return os.path.join(self.output_dir, output_filename)
