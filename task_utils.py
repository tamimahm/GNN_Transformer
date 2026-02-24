import os
import re
import pandas as pd
import numpy as np
import logging
from segment_utils import normalize_keypoints

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def normalize_patient_id(raw_id):
    """
    Normalize patient ID from various formats to integer.

    Handles formats like:
        ARAT001, ARAT_03, ARAT-004, ARAT005, arat_009,
        028, 30, 32, 100, etc.

    Returns:
        Integer patient ID or None if parsing fails
    """
    raw_str = str(raw_id).strip()
    # Remove common ARAT prefix (case-insensitive) with optional separator
    cleaned = re.sub(r'(?i)^arat[_\-]?0*', '', raw_str)
    if cleaned == '':
        # The entire string was the prefix, e.g. "ARAT" with no number
        # Try extracting digits from original
        match = re.search(r'(\d+)', raw_str)
        if match:
            return int(match.group(1))
        return None
    # Extract numeric part
    match = re.search(r'(\d+)', cleaned)
    if match:
        return int(match.group(1))
    return None


def load_live_ratings(csv_path):
    """
    Load live ratings from the clinician CSV file.

    CSV format: Patient, Activity, Rating
    Patient IDs are in mixed formats (ARAT001, ARAT_03, 028, 30, etc.)

    Args:
        csv_path: Path to the live_rating_cleaned.csv file

    Returns:
        Dictionary mapping (patient_id, activity_id) -> rating
    """
    if not os.path.exists(csv_path):
        logger.warning(f"Live ratings CSV not found: {csv_path}")
        return {}

    df = pd.read_csv(csv_path)
    live_ratings = {}
    parse_failures = 0

    for _, row in df.iterrows():
        patient_id = normalize_patient_id(row['Patient'])
        if patient_id is None:
            logger.warning(f"Could not parse patient ID: {row['Patient']}")
            parse_failures += 1
            continue
        activity_id = int(row['Activity'])
        rating = int(row['Rating'])
        live_ratings[(patient_id, activity_id)] = rating

    logger.info(f"Loaded {len(live_ratings)} live ratings from {csv_path}")
    if parse_failures > 0:
        logger.warning(f"Failed to parse {parse_failures} patient IDs from live ratings CSV")
    return live_ratings


def build_task_database(records, keypoints_data, data_type, fps=30):
    """
    Build task-level database from video records and keypoints data.

    Unlike build_segment_database, this uses the FULL activity keypoints
    without splitting by segment timing. Each entry is one complete
    (patient, activity, view) combination.

    Args:
        records: List of records from load_video_segments_info
        keypoints_data: Dictionary of keypoints organized by patient_id/activity_id
        data_type: Type of keypoints ('body', 'hand', 'object')
        fps: Frames per second of the original video

    Returns:
        Dictionary of tasks with normalized keypoints, keyed by sequential ID
    """
    tasks = {}
    task_id = 0
    seen = set()  # track (patient_id, activity_id, view_type) to avoid duplicates

    for record in records:
        patient_id = record['patient_id']
        activity_id = record['activity_id']
        camera_id = record['CameraId']
        hand_id = record['FileName'].split('_')[2]

        if (hand_id == 'left' and camera_id == 4) or (hand_id == 'right' and camera_id == 1):
            view_type = 'ipsi'
        elif camera_id == 3:
            view_type = 'top'
        else:
            view_type = 'contra'
            continue

        # Skip duplicates - for tasks we only need one entry per (patient, activity, view)
        dedup_key = (patient_id, activity_id, view_type)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        if data_type == 'object':
            if (patient_id not in keypoints_data or
                    activity_id not in keypoints_data[patient_id][view_type]):
                continue
            view_keypoints = keypoints_data[patient_id][view_type][activity_id]
            view_keypoints = view_keypoints[np.newaxis, :, :]
        else:
            if (patient_id not in keypoints_data or
                    activity_id not in keypoints_data[patient_id]):
                continue
            patient_keypoints = keypoints_data[patient_id][activity_id]
            if view_type not in patient_keypoints:
                continue
            view_keypoints = patient_keypoints[view_type]

        normalized_frames = normalize_keypoints(view_keypoints)

        tasks[task_id] = {
            'patient_id': patient_id,
            'activity_id': activity_id,
            'camera_id': camera_id,
            'view_type': view_type,
            'keypoints': normalized_frames,
            'impaired_hand': hand_id
        }
        task_id += 1

    logger.info(f"Built {len(tasks)} task entries with normalized {data_type} keypoints")
    return tasks
