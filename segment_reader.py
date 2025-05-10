import os
import pickle
import glob
import pandas as pd
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_valid_rating(segment):
    """
    Extract valid rating from segment
    
    Args:
        segment: Segment data with ratings
        
    Returns:
        Valid rating (2, 3, or "no_match") or None if no valid rating
    """
    if 'segment_ratings' not in segment:
        return None
    
    # Check t1 and t2 ratings
    t1_rating = segment['segment_ratings'].get('t1')
    t2_rating = segment['segment_ratings'].get('t2')
    
    # If both ratings are present and match
    if t1_rating is not None and t2_rating is not None and t1_rating == t2_rating:
        return t1_rating
    
    # If only one rating is present
    elif t1_rating is not None and t2_rating is None:
        return t1_rating
    elif t2_rating is not None and t1_rating is None:
        return t2_rating
    
    # If ratings don't match
    elif t1_rating is not None and t2_rating is not None and t1_rating != t2_rating:
        return "no_match"
    
    return None

def read_segment_information(pickle_dir,view_type, ipsi_contra_csv=None ):
    """
    Read segment information and therapist labels from pickle files
    
    Args:
        pickle_dir: Directory containing segment pickle files
        ipsi_contra_csv: CSV file mapping patient IDs to ipsilateral camera IDs
        camera_box: Type of camera view to use ('bboxes_top', 'bboxes_ipsi', or 'all')
        
    Returns:
        Tuple of (train_segments, inference_segments)
        - train_segments: List of segments with consensus labels for training/validation
        - inference_segments: List of segments with individual t1/t2 labels for inference
    """
    logger.info(f"Reading segment information from {pickle_dir}")
    
    # Load patient to ipsilateral camera mapping if CSV exists
    patient_to_ipsilateral = {}
    if ipsi_contra_csv and os.path.exists(ipsi_contra_csv):
        camera_df = pd.read_csv(ipsi_contra_csv)
        patient_to_ipsilateral = dict(zip(camera_df['patient_id'], camera_df['ipsilateral_camera_id']))
        logger.info(f"Loaded ipsilateral camera mapping for {len(patient_to_ipsilateral)} patients")
    
    # Find all pickle files
    pickle_files = glob.glob(os.path.join(pickle_dir, "*.pkl"))
    logger.info(f"Found {len(pickle_files)} pickle files to process.")
    
    # Statistics counters
    r2_count = 0  # Ratings of 2 (maps to label 0)
    r3_count = 0  # Ratings of 3 (maps to label 1)
    no_match_count = 0
    
    # Collect valid segments for training/validation and inference
    train_segments = []
    inference_segments = []
    
    # Process each pickle file
    for pkl_file in tqdm(pickle_files, desc="Reading segment files"):
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading pickle file {pkl_file}: {e}")
            continue
        
        # Process segments for each camera
        for camera_id in data:
            for segments_group in data[camera_id]:
                for segment in segments_group:
                    # Extract patient_id and camera_id from segment
                    patient_id = segment['patient_id']
                    segment_camera_id = segment['CameraId']
                    
                    # Determine if this is the appropriate camera view
                    if view_type == 'top' and segment_camera_id != 'cam3':
                        continue
                    elif view_type == 'ipsi' and segment_camera_id == 'cam1':
                        continue
                    
                    # For ipsilateral view, check if this camera is the ipsilateral camera for the patient
                    if view_type == 'ipsi':
                        ipsilateral_camera = patient_to_ipsilateral.get(patient_id)
                        if ipsilateral_camera != segment_camera_id:
                            continue
                    
                    # Check for valid rating
                    if 'segment_ratings' not in segment:
                        continue
                    
                    # Extract individual ratings for inference segments
                    t1_rating = segment['segment_ratings'].get('t1')
                    t2_rating = segment['segment_ratings'].get('t2')
                    
                    # Convert individual ratings to binary labels for inference segments
                    t1_label = None if t1_rating is None else (0 if t1_rating == 2 else 1)
                    t2_label = None if t2_rating is None else (0 if t2_rating == 2 else 1)
                    
                    # Create video ID for the segment
                    video_id = (f"patient_{segment['patient_id']}_task_{segment['activity_id']}_"
                              f"{segment['CameraId']}_seg_{segment['segment_id']}")
                    
                    # Determine view type and impaired hand
                    view_type = 'top' if segment_camera_id == 'cam3' else 'ipsi'
                    impaired_hand = 1 if 'left_Impaired' in video_id else 0  # 0 for right, 1 for left
                    
                    # Add to inference_segments (with t1_label and t2_label)
                    inference_segments.append({
                        'frames': segment['frames'],
                        'video_id': video_id,
                        't1_label': t1_label,
                        't2_label': t2_label,
                        'camera_id': segment_camera_id,
                        'view_type': view_type,
                        'patient_id': patient_id,
                        'activity_id': segment['activity_id'],
                        'segment_id': segment['segment_id'],
                        'impaired_hand': impaired_hand
                    })
                    
                    # For training segments, we need a consensus label
                    rating = get_valid_rating(segment)
                    
                    # Skip segments with no rating or rating not in [2, 3]
                    if rating is None or (rating != "no_match" and rating not in [2, 3]):
                        continue
                    
                    # Handle matching ratings
                    if rating == "no_match":
                        no_match_count += 1
                        continue
                    
                    # Convert rating to label
                    try:
                        rating = int(rating)
                        
                        # Map according to specified scheme: 2->0, 3->1
                        if rating == 2:
                            label = 0
                            r2_count += 1
                        else:  # rating == 3
                            label = 1
                            r3_count += 1
                    except (ValueError, TypeError):
                        logger.warning(f"Skipping segment in {pkl_file} with invalid rating: {rating}")
                        continue
                    
                    # Add to train_segments (for segments with matching ratings)
                    train_segments.append({
                        'frames': segment['frames'],
                        'video_id': video_id,
                        'label': label,
                        'camera_id': segment_camera_id,
                        'view_type': view_type,
                        'patient_id': patient_id,
                        'activity_id': segment['activity_id'],
                        'segment_id': segment['segment_id'],
                        'impaired_hand': impaired_hand
                    })
    
    # Log statistics
    logger.info(f"Read {len(train_segments)} segments with valid therapist ratings for training")
    logger.info(f"  Class 0 (rating 2): {r2_count} segments")
    logger.info(f"  Class 1 (rating 3): {r3_count} segments")
    logger.info(f"  No match: {no_match_count} segments")
    logger.info(f"Read {len(inference_segments)} segments for inference")
    
    return train_segments, inference_segments