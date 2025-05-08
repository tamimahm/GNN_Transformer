import os
import pickle
import glob
import numpy as np
from tqdm import tqdm
import pandas as pd

class MultiPickleProcessor:
    """
    Class to process multiple pickle directories and create a combined segment database
    for the GNN + Transformer model
    """
    
    def __init__(self, 
                pickle_dirs={
                    'body': 'D:/pickle_dir',            # Body keypoints from OpenPose
                    'openpose': 'D:/pickle_files',      # Alternative OpenPose data format
                    'hand': 'D:/pickle_files_hand',     # Hand keypoints from MediaPipe
                    'object': 'D:/pickle_files_object'  # Object locations from TridentNet
                },
                output_dir='D:/combined_segments',
                num_files_per_dir=10):
        """
        Initialize the processor with pickle directories
        
        Args:
            pickle_dirs: Dictionary of pickle directories (key is type, value is directory path)
            output_dir: Directory to save the combined segment database
            num_files_per_dir: Number of pickle files to read from each directory
        """
        self.pickle_dirs = pickle_dirs
        self.output_dir = output_dir
        self.num_files_per_dir = num_files_per_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize data containers
        self.body_data = {}      # Patient ID -> activity ID -> keypoints
        self.hand_data = {}      # Patient ID -> activity ID -> keypoints
        self.object_data = {}    # Patient ID -> activity ID -> object locations
        self.segment_data = {}   # Segment ID -> combined data
    
    def load_pickle_files(self):
        """
        Load pickle files from each directory
        """
        print("Loading pickle files...")
        
        # Load body keypoints (from pickle_dir)
        if 'body' in self.pickle_dirs:
            self._load_body_pickle_files()
            
        # Load alternative OpenPose format (from pickle_files)
        if 'openpose' in self.pickle_dirs:
            self._load_openpose_pickle_files()
            
        # Load hand keypoints (from pickle_files_hand)
        if 'hand' in self.pickle_dirs:
            self._load_hand_pickle_files()
            
        # Load object locations (from pickle_files_object)
        if 'object' in self.pickle_dirs:
            self._load_object_pickle_files()
    
    def _load_body_pickle_files(self):
        """
        Load body keypoint pickle files (from pickle_dir)
        """
        body_dir = self.pickle_dirs.get('body')
        if not body_dir or not os.path.exists(body_dir):
            print(f"Body keypoints directory not found: {body_dir}")
            return
            
        # Find all pickle files
        pickle_files = glob.glob(os.path.join(body_dir, "*.pkl"))
        if len(pickle_files) == 0:
            print(f"No pickle files found in {body_dir}")
            return
            
        # Limit to specified number of files
        pickle_files = pickle_files[:self.num_files_per_dir]
        
        print(f"Loading {len(pickle_files)} body keypoint files...")
        for pkl_file in tqdm(pickle_files):
            try:
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                
                # Process based on expected format
                # Assuming data is dict with patient IDs as keys
                for patient_id, patient_data in data.items():
                    if patient_id not in self.body_data:
                        self.body_data[patient_id] = {}
                    
                    # Check the structure (might be different between files)
                    if isinstance(patient_data, dict) and 'activities' in patient_data:
                        # Format from the improved processor
                        for activity_id, activity_data in patient_data['activities'].items():
                            self.body_data[patient_id][activity_id] = activity_data
                    elif isinstance(patient_data, dict) and ('top' in patient_data or 'ipsi' in patient_data):
                        # Format from the updated processor
                        self.body_data[patient_id] = patient_data
                    else:
                        # Original format (list of file info)
                        for file_info in patient_data:
                            if 'file_path' in file_info and 'data' in file_info:
                                # Extract activity ID from file path
                                file_path = file_info['file_path']
                                activity_id = self._extract_activity_id_from_path(file_path)
                                
                                if activity_id:
                                    if activity_id not in self.body_data[patient_id]:
                                        self.body_data[patient_id][activity_id] = []
                                    
                                    self.body_data[patient_id][activity_id].append(file_info['data'])
            except Exception as e:
                print(f"Error loading {pkl_file}: {e}")
    
    def _load_openpose_pickle_files(self):
        """
        Load OpenPose pickle files (from pickle_files)
        """
        openpose_dir = self.pickle_dirs.get('openpose')
        if not openpose_dir or not os.path.exists(openpose_dir):
            print(f"OpenPose directory not found: {openpose_dir}")
            return
            
        # Find all pickle files
        pickle_files = glob.glob(os.path.join(openpose_dir, "*.pkl"))
        if len(pickle_files) == 0:
            print(f"No pickle files found in {openpose_dir}")
            return
            
        # Limit to specified number of files
        pickle_files = pickle_files[:self.num_files_per_dir]
        
        print(f"Loading {len(pickle_files)} OpenPose files...")
        for pkl_file in tqdm(pickle_files):
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
                                
                                if 'views' not in self.body_data[patient_id][activity_id]:
                                    self.body_data[patient_id][activity_id]['views'] = {}
                                
                                if view_type not in self.body_data[patient_id][activity_id]['views']:
                                    self.body_data[patient_id][activity_id]['views'][view_type] = {}
                                
                                self.body_data[patient_id][activity_id]['views'][view_type]['keypoints'] = keypoints
            except Exception as e:
                print(f"Error loading {pkl_file}: {e}")
    
    def _load_hand_pickle_files(self):
        """
        Load hand keypoint pickle files (from pickle_files_hand)
        """
        hand_dir = self.pickle_dirs.get('hand')
        if not hand_dir or not os.path.exists(hand_dir):
            print(f"Hand keypoints directory not found: {hand_dir}")
            return
            
        # Find all pickle files
        pickle_files = glob.glob(os.path.join(hand_dir, "*.pkl"))
        if len(pickle_files) == 0:
            print(f"No pickle files found in {hand_dir}")
            return
            
        # Limit to specified number of files
        pickle_files = pickle_files[:self.num_files_per_dir]
        
        print(f"Loading {len(pickle_files)} hand keypoint files...")
        for pkl_file in tqdm(pickle_files):
            try:
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                
                # Process based on expected format
                for patient_id, patient_data in data.items():
                    if patient_id not in self.hand_data:
                        self.hand_data[patient_id] = {}
                    
                    # Assume similar format to body data
                    # This might need adjustment based on actual hand data format
                    if 'top' in patient_data and 'ipsi' in patient_data:
                        for view_type in ['top', 'ipsi']:
                            for activity_id, keypoints in patient_data[view_type].items():
                                if activity_id not in self.hand_data[patient_id]:
                                    self.hand_data[patient_id][activity_id] = {}
                                
                                if view_type not in self.hand_data[patient_id][activity_id]:
                                    self.hand_data[patient_id][activity_id][view_type] = {}
                                
                                self.hand_data[patient_id][activity_id][view_type] = keypoints
            except Exception as e:
                print(f"Error loading {pkl_file}: {e}")
    
    def _load_object_pickle_files(self):
        """
        Load object location pickle files (from pickle_files_object)
        """
        object_dir = self.pickle_dirs.get('object')
        if not object_dir or not os.path.exists(object_dir):
            print(f"Object locations directory not found: {object_dir}")
            return
            
        # Find all pickle files
        pickle_files = glob.glob(os.path.join(object_dir, "*.pkl"))
        if len(pickle_files) == 0:
            print(f"No pickle files found in {object_dir}")
            return
            
        # Limit to specified number of files
        pickle_files = pickle_files[:self.num_files_per_dir]
        
        print(f"Loading {len(pickle_files)} object location files...")
        for pkl_file in tqdm(pickle_files):
            try:
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                
                # Process based on expected format
                for patient_id, patient_data in data.items():
                    if patient_id not in self.object_data:
                        self.object_data[patient_id] = {}
                    
                    # Assume similar format to other data
                    # This might need adjustment based on actual object data format
                    if isinstance(patient_data, dict):
                        for activity_id, object_info in patient_data.items():
                            self.object_data[patient_id][activity_id] = object_info
            except Exception as e:
                print(f"Error loading {pkl_file}: {e}")
    
    def _extract_activity_id_from_path(self, file_path):
        """
        Extract activity ID from file path
        """
        try:
            # Look for "activity_" in the path
            if 'activity_' in file_path:
                parts = file_path.split('activity_')
                if len(parts) > 1:
                    # Extract the numeric part after "activity_"
                    activity_id_part = parts[1].split(os.sep)[0]
                    return int(activity_id_part)
            return None
        except (ValueError, IndexError):
            return None
    
    def build_segment_database(self, therapist_labels_path=None):
        """
        Build segment database by combining body, hand, and object data
        
        Args:
            therapist_labels_path: Path to CSV or pickle file with therapist labels
        """
        print("Building segment database...")
        
        # Load therapist labels if provided
        therapist_labels = None
        if therapist_labels_path and os.path.exists(therapist_labels_path):
            if therapist_labels_path.endswith('.csv'):
                therapist_labels = pd.read_csv(therapist_labels_path)
            elif therapist_labels_path.endswith('.pkl'):
                with open(therapist_labels_path, 'rb') as f:
                    therapist_labels = pickle.load(f)
        
        # Create segments by combining data from different sources
        segment_id = 0
        for patient_id in self.body_data.keys():
            print(f"Processing patient {patient_id}...")
            
            for activity_id in self.body_data[patient_id].keys():
                # Get body data for this activity
                body_keypoints = self._get_body_keypoints(patient_id, activity_id)
                
                if body_keypoints is None:
                    print(f"No body keypoints found for patient {patient_id}, activity {activity_id}")
                    continue
                
                # Get hand data for this activity
                hand_keypoints = self._get_hand_keypoints(patient_id, activity_id)
                
                # Get object data for this activity
                object_locations = self._get_object_locations(patient_id, activity_id)
                
                # Get labels for this activity
                labels = self._get_therapist_labels(therapist_labels, patient_id, activity_id)
                
                # Create segments (this is a simplified version - adjust based on actual data structure)
                # In a real scenario, you'd need to align the frames between different modalities
                for view_type in ['top', 'ipsi']:
                    if view_type not in body_keypoints:
                        continue
                    
                    segment = {
                        'patient_id': patient_id,
                        'activity_id': activity_id,
                        'view_type': view_type,
                        'body_keypoints': body_keypoints[view_type],
                        'hand_keypoints': hand_keypoints.get(view_type) if hand_keypoints else None,
                        'object_locations': object_locations,
                        'label': labels.get('score') if labels else None
                    }
                    
                    # Store the segment
                    self.segment_data[segment_id] = segment
                    segment_id += 1
        
        print(f"Created {segment_id} segments")
    
    def _get_body_keypoints(self, patient_id, activity_id):
        """
        Get body keypoints for a specific patient and activity
        """
        if patient_id not in self.body_data or activity_id not in self.body_data[patient_id]:
            return None
        
        activity_data = self.body_data[patient_id][activity_id]
        
        # Different format handling
        if 'views' in activity_data:
            return activity_data['views']
        elif isinstance(activity_data, dict) and ('top' in activity_data or 'ipsi' in activity_data):
            return activity_data
        else:
            # Might need to process raw data into the required format
            # This is a placeholder - adjust based on actual data structure
            return None
    
    def _get_hand_keypoints(self, patient_id, activity_id):
        """
        Get hand keypoints for a specific patient and activity
        """
        if patient_id not in self.hand_data or activity_id not in self.hand_data[patient_id]:
            return None
        
        return self.hand_data[patient_id][activity_id]
    
    def _get_object_locations(self, patient_id, activity_id):
        """
        Get object locations for a specific patient and activity
        """
        if patient_id not in self.object_data or activity_id not in self.object_data[patient_id]:
            return None
        
        return self.object_data[patient_id][activity_id]
    
    def _get_therapist_labels(self, therapist_labels, patient_id, activity_id):
        """
        Get therapist labels for a specific patient and activity
        """
        if therapist_labels is None:
            return None
        
        # This is a placeholder - adjust based on actual label structure
        if isinstance(therapist_labels, pd.DataFrame):
            # Filter DataFrame
            filtered = therapist_labels[(therapist_labels['patient_id'] == patient_id) & 
                                       (therapist_labels['activity_id'] == activity_id)]
            if len(filtered) > 0:
                return filtered.iloc[0].to_dict()
        elif isinstance(therapist_labels, dict):
            # Check if patient ID and activity ID are in the dictionary
            if patient_id in therapist_labels and activity_id in therapist_labels[patient_id]:
                return therapist_labels[patient_id][activity_id]
        
        return None
    
    def save_segment_database(self, filename='segment_database.pkl'):
        """
        Save the segment database
        """
        output_path = os.path.join(self.output_dir, filename)
        print(f"Saving segment database to {output_path}...")
        
        with open(output_path, 'wb') as f:
            pickle.dump(self.segment_data, f)
        
        print(f"Saved {len(self.segment_data)} segments")
    
    def process(self, therapist_labels_path=None, output_filename='segment_database.pkl'):
        """
        Run the full processing pipeline
        
        Args:
            therapist_labels_path: Path to CSV or pickle file with therapist labels
            output_filename: Filename for the output segment database
        """
        self.load_pickle_files()
        self.build_segment_database(therapist_labels_path)
        self.save_segment_database(output_filename)

if __name__ == "__main__":
    # Example usage
    processor = MultiPickleProcessor(
        pickle_dirs={
            'body': 'D:/pickle_dir',            # Body keypoints from OpenPose
            'openpose': 'D:/pickle_files',      # Alternative OpenPose data format
            'hand': 'D:/pickle_files_hand',     # Hand keypoints from MediaPipe
            'object': 'D:/pickle_files_object'  # Object locations from TridentNet
        },
        output_dir='D:/combined_segments',
        num_files_per_dir=10
    )
    
    # Full process with therapist labels
    processor.process(therapist_labels_path='D:/pickle_dir/therapist_labels.csv')
