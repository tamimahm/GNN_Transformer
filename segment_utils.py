import os
import pandas as pd
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_video_segments_info(csv_dir):
    """
    Loads and merges the updated CSV files.

    Expected files in csv_dir:
      - pth_updated.csv with columns: FileName, PatientTaskHandmappingId, CameraId
      - segmentation_updated.csv with columns: PatientTaskHandMappingId, SegmentId, Start, End

    Returns:
      A list of dictionaries. Each dictionary corresponds to one video file record with keys:
         'FileName', 'PatientTaskHandmappingId', 'CameraId', 'patient_id', 'activity_id', 'segments'
      where 'segments' is a list of (start, end) tuples.
    """
    pth_file = os.path.join(csv_dir, "pth_updated.csv")
    seg_file = os.path.join(csv_dir, "segmentation_updated.csv")
    
    # Read CSVs
    pth_df = pd.read_csv(pth_file)
    seg_df = pd.read_csv(seg_file)
    
    # Merge based on PatientTaskHandmappingId (note: column names differ in case)
    merged_df = pd.merge(pth_df, seg_df, left_on='PatientTaskHandmappingId', right_on='PatientTaskHandMappingId')
    
    # Group by FileName, PatientTaskHandmappingId, and CameraId to aggregate segments
    grouped = merged_df.groupby(['FileName', 'PatientTaskHandmappingId', 'CameraId'])
    
    records = []
    # Define activities to skip
    skip_activities = {"7", "17", "18", "19"}
    for (file_name, mapping_id, camera_id), group in grouped:
        # Aggregate segments as list of tuples (start, end)
        segments = list(zip(group['Start'], group['End']))
        # Split filename to extract patient id and activity id.
        # Expected filename format: ARAT_01_right_Impaired_cam1_activity11.mp4
        parts = file_name.split("_")
        if len(parts) < 5:
            logger.warning(f"Filename {file_name} does not match expected format. Skipping.")
            continue
        # Assume patient id is the numeric part from the second token, e.g., "01" from "ARAT_01"
        patient_id = int(parts[1].strip())
        # Activity part is assumed to be the last component (e.g., "activity11.mp4")
        activity_part = parts[-1]
        activity_id = activity_part.split('.')[0].replace("activity", "").strip()
        # Skip specified activities
        if activity_id in skip_activities:
            continue        
        record = {
            "FileName": file_name,
            "PatientTaskHandmappingId": mapping_id,
            "CameraId": int(camera_id),  # e.g., 1,2,3, etc.
            "patient_id": patient_id,
            "activity_id": int(activity_id),
            "segments": segments
        }
        records.append(record)
    
    return records

def extract_segment_frames(keypoints_array, fps, start_time, end_time, num_target_frames=20):
    """
    Extract frames from a keypoints array for a specific time segment
    
    Args:
        keypoints_array: Array of keypoints with shape (keypoints, coords, frames)
        fps: Frames per second of the original video
        start_time: Start time of the segment in seconds
        end_time: End time of the segment in seconds
        num_target_frames: Number of frames to extract from the segment
    
    Returns:
        Array of keypoints for the specified segment with exactly num_target_frames
    """
    # Calculate start and end frame indices
    start_frame = int(start_time)
    end_frame = int(end_time )
    
    # Ensure we don't exceed array bounds
    total_keypoint_frames = keypoints_array.shape[2]
    start_frame = min(start_frame, 0)
    # If end_frame exceeds total frames, adjust it
    if end_frame > total_keypoint_frames:
        end_frame = total_keypoint_frames        
    # Ensure start_frame is less than end_frame
    if start_frame >= end_frame:
        logger.warning(f"Invalid segment: start_frame ({start_frame}) >= end_frame ({end_frame})")
        end_frame = start_frame + 1
        return []
    
    # Extract frames for the segment
    segment_frames = keypoints_array[:, :, start_frame:end_frame]
    
    # Determine number of frames in the segment
    num_segment_frames = segment_frames.shape[2]
    
    # If we have fewer frames than needed, duplicate the last frame
    if num_segment_frames < num_target_frames:
        last_frame = segment_frames[:, :, -1]
        padding_needed = num_target_frames - num_segment_frames
        padding = np.repeat(last_frame[:, :, np.newaxis], padding_needed, axis=2)
        segment_frames = np.concatenate([segment_frames, padding], axis=2)
        num_segment_frames = num_target_frames
    
    # Get evenly spaced indices
    if num_segment_frames == num_target_frames:
        frame_indices = list(range(num_segment_frames))
    else:
        frame_indices = np.linspace(0, num_segment_frames - 1, num_target_frames, dtype=int)
    
    # Extract the target frames
    target_frames = segment_frames[:, :, frame_indices].copy()
    
    return target_frames

def normalize_keypoints(keypoints, image_width=1920, image_height=1080):
    """
    Normalize keypoint coordinates to 0-1 range
    
    Args:
        keypoints: Array of keypoints with shape (keypoints, coords, frames)
        image_width: Width of the original image
        image_height: Height of the original image
    
    Returns:
        Normalized keypoints
    """
    normalized = np.zeros([keypoints.shape[0],2,keypoints.shape[2]])
    
    # Check if already normalized (max values are <= 1.0)
    x_max = np.max(keypoints[:, 0, :]) if keypoints.size > 0 else 0
    y_max = np.max(keypoints[:, 1, :]) if keypoints.size > 0 else 0
    
    if x_max > 1.0 or y_max > 1.0:
        # Normalize X coordinates (index 0) by image width
        for kp in range(keypoints.shape[0]):
            normalized[kp, 0, :] = keypoints[kp, 0, :] / image_width
    
            # Normalize Y coordinates (index 1) by image height
            normalized[kp, 1, :] = keypoints[kp, 1, :] / image_height
    
        # Log the normalization
        logger.info(f"Normalized keypoints from max values (X: {x_max}, Y: {y_max}) to 0-1 range")
    else:
        normalized=keypoints
        logger.info("Keypoints appear to be already normalized (values <= 1.0)")

    
    return normalized

def build_segment_database(records, keypoints_data, data_type,fps=30, num_target_frames=20):
    """
    Build segment database from video segment records and keypoints data
    
    Args:
        records: List of segment records from load_video_segments_info
        keypoints_data: Dictionary of keypoints data, organized by patient_id and activity_id
        fps: Frames per second of the original video
        num_target_frames: Number of frames to extract per segment
    
    Returns:
        Dictionary of segments with normalized keypoints
    """
    segments = {}
    segment_id = 0
    
    for record in records:
        patient_id = record['patient_id']
        activity_id = record['activity_id']
        camera_id = record['CameraId']
        hand_id=record['FileName'].split('_')[2]
        if (hand_id=='left' and camera_id=='4') or (hand_id=='right' and camera_id=='1'):
            view_type='ipsi'
        elif camera_id==3:
            view_type='top'
        else:
            view_type='contra'
            continue
            
        if data_type=='object':
            # Skip if no keypoints data for this patient/activity
            if (patient_id not in keypoints_data or 
                activity_id not in keypoints_data[patient_id][view_type]):
                continue
            else:
                view_keypoints = keypoints_data[patient_id][view_type][activity_id]
                view_keypoints=view_keypoints[np.newaxis,:,:]
        else:
            # Skip if no keypoints data for this patient/activity
            if (patient_id not in keypoints_data or 
                activity_id not in keypoints_data[patient_id]):
                continue
            else:
                patient_keypoints = keypoints_data[patient_id][activity_id]            
                # Skip if no keypoints for this view
                if view_type not in patient_keypoints:
                    continue
                # Get keypoints for this view
                view_keypoints = patient_keypoints[view_type]
        
        # Process each segment
        for seg_index, seg in enumerate(record['segments']):
            start_time, end_time = seg
            
            # Extract segment frames
            try:
                segment_frames = extract_segment_frames(
                    view_keypoints, fps, start_time, end_time, num_target_frames
                )
                
                # Normalize keypoints
                normalized_frames = normalize_keypoints(segment_frames)
                
                # Create segment
                segments[segment_id] = {
                    'patient_id': patient_id,
                    'activity_id': activity_id,
                    'segment_id': seg_index,
                    'camera_id': camera_id,
                    'view_type': view_type,
                    'start_time': start_time,
                    'end_time': end_time,
                    'keypoints': normalized_frames
                }
                
                segment_id += 1
            except Exception as e:
                logger.error(f"Error processing segment {seg_index} for patient {patient_id}, activity {activity_id}: {e}")
    
    logger.info(f"Built {len(segments)} segments with normalized keypoints")
    return segments