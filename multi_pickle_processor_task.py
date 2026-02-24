import os
import pickle
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
from task_reader import read_task_information
from task_utils import load_live_ratings, build_task_database
from segment_utils import load_video_segments_info

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultiPickleTaskProcessor:
    """
    Class to process multiple pickle directories and create a combined task-level database
    for the GNN + Transformer model.

    Unlike the segment processor, this operates at the task (activity) level:
    - Uses task_ratings from task pickle files (not segment_ratings)
    - Uses full activity keypoints (no segment timing extraction)
    - Adds live_rating from clinician CSV when available
    """

    def __init__(self,
                pickle_dirs={
                    'pickle_dir': 'D:/nature_everything/nature_dataset/task_dataset',
                    'openpose': 'D:/pickle_files',
                    'hand': 'D:/pickle_files_hand',
                    'object': 'D:/pickle_files_object'
                },
                csv_dir='D:/nature_everything',
                output_dir='D:/nature_everything/combined_tasks',
                ipsi_contra_csv='D:/nature_everything/camera_assignments.csv',
                live_rating_csv=None,
                num_files_per_dir=10,
                fps=30,
                view_type='top'):
        """
        Initialize the task processor.

        Args:
            pickle_dirs: Dictionary of pickle directories
            csv_dir: Directory with segment timing CSV files (for keypoint record lookup)
            output_dir: Directory to save the combined task database
            ipsi_contra_csv: CSV mapping patient IDs to ipsilateral camera IDs
            live_rating_csv: Path to live_rating_cleaned.csv with clinician live ratings
            num_files_per_dir: Number of pickle files to read from each directory
            fps: Frames per second of the original videos
            view_type: Camera view type ('top' or 'ipsi')
        """
        self.pickle_dirs = pickle_dirs
        self.csv_dir = csv_dir
        self.output_dir = output_dir
        self.ipsi_contra_csv = ipsi_contra_csv
        self.live_rating_csv = live_rating_csv
        self.num_files_per_dir = num_files_per_dir
        self.fps = fps
        self.view_type = view_type

        os.makedirs(output_dir, exist_ok=True)

        # Data containers
        self.body_data = {}
        self.hand_data = {}
        self.object_data = {}
        self.train_task_data = {}
        self.inference_task_data = {}
        self.video_records = None
        self.live_ratings = {}

    def load_video_records(self):
        """
        Load video records from CSV files.
        These records map (patient_id, activity_id) to camera/view info
        needed to look up the correct keypoint data.
        """
        logger.info("Loading video records for keypoint lookup...")
        self.video_records = load_video_segments_info(self.csv_dir)
        logger.info(f"Loaded records for {len(self.video_records)} video files")
        return self.video_records

    def load_live_ratings(self):
        """Load live ratings from clinician CSV."""
        if self.live_rating_csv:
            self.live_ratings = load_live_ratings(self.live_rating_csv)
        else:
            logger.info("No live rating CSV provided, skipping live ratings")
            self.live_ratings = {}
        return self.live_ratings

    def load_pickle_files(self):
        """Load keypoint pickle files from each directory."""
        logger.info("Loading keypoint pickle files...")

        if 'openpose' in self.pickle_dirs:
            self._load_openpose_pickle_files()
        if 'hand' in self.pickle_dirs:
            self._load_hand_pickle_files()
        if 'object' in self.pickle_dirs:
            self._load_object_pickle_files()

    def _load_openpose_pickle_files(self):
        """Load OpenPose body keypoint pickle files."""
        openpose_dir = self.pickle_dirs.get('openpose')
        if not openpose_dir or not os.path.exists(openpose_dir):
            logger.warning(f"OpenPose directory not found: {openpose_dir}")
            return

        pickle_files = glob.glob(os.path.join(openpose_dir, "*.pkl"))
        if len(pickle_files) == 0:
            logger.warning(f"No pickle files found in {openpose_dir}")
            return

        pickle_files = pickle_files[:self.num_files_per_dir]

        logger.info(f"Loading {len(pickle_files)} OpenPose files...")
        for pkl_file in tqdm(pickle_files, desc="Loading OpenPose files"):
            try:
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)

                for patient_id, patient_data in data.items():
                    if patient_id not in self.body_data:
                        self.body_data[patient_id] = {}

                    if 'top' in patient_data and 'ipsi' in patient_data:
                        for view_type in ['top', 'ipsi']:
                            for activity_id, keypoints in patient_data[view_type].items():
                                if activity_id not in self.body_data[patient_id]:
                                    self.body_data[patient_id][activity_id] = {}
                                self.body_data[patient_id][activity_id][view_type] = keypoints
            except Exception as e:
                logger.error(f"Error loading {pkl_file}: {e}")

    def _load_hand_pickle_files(self):
        """Load hand keypoint pickle files."""
        hand_dir = self.pickle_dirs.get('hand')
        if not hand_dir or not os.path.exists(hand_dir):
            logger.warning(f"Hand keypoints directory not found: {hand_dir}")
            return

        pickle_files = glob.glob(os.path.join(hand_dir, "*.pkl"))
        if len(pickle_files) == 0:
            logger.warning(f"No pickle files found in {hand_dir}")
            return

        pickle_files = pickle_files[:self.num_files_per_dir]

        logger.info(f"Loading {len(pickle_files)} hand keypoint files...")
        for pkl_file in tqdm(pickle_files, desc="Loading hand keypoint files"):
            try:
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)

                for patient_id, patient_data in data.items():
                    if patient_id not in self.hand_data:
                        self.hand_data[patient_id] = {}

                    if 'top' in patient_data and 'ipsi' in patient_data:
                        for view_type in ['top', 'ipsi']:
                            for activity_id, keypoints in patient_data[view_type].items():
                                if activity_id not in self.hand_data[patient_id]:
                                    self.hand_data[patient_id][activity_id] = {}
                                self.hand_data[patient_id][activity_id][view_type] = keypoints
            except Exception as e:
                logger.error(f"Error loading {pkl_file}: {e}")

    def _load_object_pickle_files(self):
        """Load object location pickle files."""
        object_dir = self.pickle_dirs.get('object')
        if not object_dir or not os.path.exists(object_dir):
            logger.warning(f"Object locations directory not found: {object_dir}")
            return

        pickle_files = glob.glob(os.path.join(object_dir, "*.pkl"))
        if len(pickle_files) == 0:
            logger.warning(f"No pickle files found in {object_dir}")
            return

        pickle_files = pickle_files[:self.num_files_per_dir]

        logger.info(f"Loading {len(pickle_files)} object location files...")
        for pkl_file in tqdm(pickle_files, desc="Loading object location files"):
            try:
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)

                for patient_id, patient_data in data.items():
                    if patient_id not in self.object_data:
                        self.object_data[patient_id] = {}

                    if isinstance(patient_data, dict):
                        for activity_id, object_info in patient_data.items():
                            self.object_data[patient_id][activity_id] = object_info
            except Exception as e:
                logger.error(f"Error loading {pkl_file}: {e}")

    def load_task_information(self):
        """
        Load task information with therapist labels from task pickle files.
        Uses task_reader which handles task_ratings properly.

        Returns:
            Tuple of (train_tasks, inference_tasks)
        """
        pickle_dir = self.pickle_dirs.get('pickle_dir')
        if not pickle_dir or not os.path.exists(pickle_dir):
            logger.error(f"Task pickle directory not found: {pickle_dir}")
            return [], []

        train_tasks, inference_tasks = read_task_information(
            pickle_dir=pickle_dir,
            view_type=self.view_type,
            ipsi_contra_csv=self.ipsi_contra_csv
        )

        logger.info(f"Loaded {len(train_tasks)} training tasks and {len(inference_tasks)} inference tasks")
        return train_tasks, inference_tasks

    def build_task_databases(self, view_types=['top', 'ipsi']):
        """
        Build training and inference task databases.

        Uses full activity keypoints (no segment splitting) and matches
        them with task_ratings from the task pickle files.

        Args:
            view_types: List of view types to include

        Returns:
            Tuple of (train_task_data, inference_task_data)
        """
        logger.info("Building task databases...")

        if self.video_records is None:
            self.load_video_records()

        # Load task labels from task pickle files
        train_tasks, inference_tasks = self.load_task_information()

        # Build task-level keypoint databases (full activity, no segment splitting)
        body_tasks = build_task_database(
            self.video_records,
            self.body_data,
            data_type='body',
            fps=self.fps
        )

        hand_tasks = {}
        if self.hand_data:
            hand_tasks = build_task_database(
                self.video_records,
                self.hand_data,
                data_type='hand',
                fps=self.fps
            )

        object_tasks = {}
        if self.object_data:
            object_tasks = build_task_database(
                self.video_records,
                self.object_data,
                data_type='object',
                fps=self.fps
            )

        # Build training task database
        self._build_training_tasks(
            train_tasks,
            body_tasks,
            hand_tasks,
            object_tasks,
            view_types
        )

        # Build inference task database
        self._build_inference_tasks(
            inference_tasks,
            body_tasks,
            hand_tasks,
            object_tasks,
            view_types
        )

        return self.train_task_data, self.inference_task_data

    def _build_training_tasks(self, train_tasks, body_tasks, hand_tasks, object_tasks, view_types):
        """
        Build training task database with consensus labels and live ratings.

        Maps task labels by (patient_id, activity_id) since tasks have no segment_id.
        """
        task_id = 0

        # Create label map: (patient_id, activity_id) -> label
        label_map = {}
        for task in train_tasks:
            key = (task['patient_id'], task['activity_id'])
            label_map[key] = task['label']

        # Process each body task entry
        for key, task_entry in body_tasks.items():
            patient_id = task_entry['patient_id']
            activity_id = task_entry['activity_id']
            view_type = task_entry['view_type']

            if view_types and view_type not in view_types:
                continue

            # Get label from task_ratings consensus
            rating_key = (patient_id, activity_id)
            label = label_map.get(rating_key)

            if label is None:
                continue

            # Get hand keypoints if available
            hand_keypoints = None
            if hand_tasks and key in hand_tasks:
                hand_keypoints = hand_tasks[key]['keypoints']

            # Get object locations if available
            object_locations = None
            if object_tasks and key in object_tasks:
                object_locations = object_tasks[key]['keypoints']

            # Get live rating if available
            live_rating = self.live_ratings.get((patient_id, activity_id), None)

            self.train_task_data[task_id] = {
                'patient_id': patient_id,
                'activity_id': activity_id,
                'camera_id': task_entry['camera_id'],
                'view_type': view_type,
                'body_keypoints': task_entry['keypoints'],
                'hand_keypoints': hand_keypoints,
                'object_locations': object_locations,
                'label': label,
                'live_rating': live_rating,
                'impaired_hand': task_entry['impaired_hand']
            }

            task_id += 1

        live_count = sum(1 for v in self.train_task_data.values() if v['live_rating'] is not None)
        logger.info(f"Built {len(self.train_task_data)} training tasks with consensus labels "
                     f"({live_count} have live ratings)")

    def _build_inference_tasks(self, inference_tasks, body_tasks, hand_tasks, object_tasks, view_types):
        """
        Build inference task database with individual t1/t2 labels and live ratings.

        Maps task labels by (patient_id, activity_id) since tasks have no segment_id.
        """
        task_id = 0

        # Create label maps: (patient_id, activity_id) -> label
        t1_label_map = {}
        t2_label_map = {}
        for task in inference_tasks:
            key = (task['patient_id'], task['activity_id'])
            t1_label_map[key] = task['t1_label']
            t2_label_map[key] = task['t2_label']

        # Process each body task entry
        for key, task_entry in body_tasks.items():
            patient_id = task_entry['patient_id']
            activity_id = task_entry['activity_id']
            view_type = task_entry['view_type']

            if view_types and view_type not in view_types:
                continue

            # Get t1 and t2 labels
            rating_key = (patient_id, activity_id)
            t1_label = t1_label_map.get(rating_key)
            t2_label = t2_label_map.get(rating_key)

            if t1_label is None and t2_label is None:
                continue

            # Get hand keypoints if available
            hand_keypoints = None
            if hand_tasks and key in hand_tasks:
                hand_keypoints = hand_tasks[key]['keypoints']

            # Get object locations if available
            object_locations = None
            if object_tasks and key in object_tasks:
                object_locations = object_tasks[key]['keypoints']

            # Get live rating if available
            live_rating = self.live_ratings.get((patient_id, activity_id), None)

            self.inference_task_data[task_id] = {
                'patient_id': patient_id,
                'activity_id': activity_id,
                'camera_id': task_entry['camera_id'],
                'view_type': view_type,
                'body_keypoints': task_entry['keypoints'],
                'hand_keypoints': hand_keypoints,
                'object_locations': object_locations,
                't1_label': t1_label,
                't2_label': t2_label,
                'live_rating': live_rating,
                'impaired_hand': task_entry['impaired_hand']
            }

            task_id += 1

        live_count = sum(1 for v in self.inference_task_data.values() if v['live_rating'] is not None)
        logger.info(f"Built {len(self.inference_task_data)} inference tasks with t1/t2 labels "
                     f"({live_count} have live ratings)")

    def save_task_databases(self):
        """Save the task databases to pickle files."""
        if self.view_type == 'top':
            train_filename = 'train_task_database_top.pkl'
            inference_filename = 'inference_task_database_top.pkl'
        else:
            train_filename = 'train_task_database_ipsi.pkl'
            inference_filename = 'inference_task_database_ipsi.pkl'

        train_output_path = os.path.join(self.output_dir, train_filename)
        logger.info(f"Saving training task database to {train_output_path}...")
        with open(train_output_path, 'wb') as f:
            pickle.dump(self.train_task_data, f)
        logger.info(f"Saved {len(self.train_task_data)} training tasks")

        inference_output_path = os.path.join(self.output_dir, inference_filename)
        logger.info(f"Saving inference task database to {inference_output_path}...")
        with open(inference_output_path, 'wb') as f:
            pickle.dump(self.inference_task_data, f)
        logger.info(f"Saved {len(self.inference_task_data)} inference tasks")

        return train_output_path, inference_output_path

    def process(self, view_types=['top', 'ipsi']):
        """
        Run the full task processing pipeline.

        Args:
            view_types: List of view types to include ('top', 'ipsi', or both)

        Returns:
            Dictionary with paths to saved task databases
        """
        self.load_video_records()
        self.load_live_ratings()
        self.load_pickle_files()
        self.build_task_databases(view_types)
        train_path, inference_path = self.save_task_databases()

        return {
            'train_task_db_path': train_path,
            'inference_task_db_path': inference_path
        }
