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


def get_valid_task_rating(task):
    """
    Extract valid rating from task's task_ratings.

    Args:
        task: Task data dict with 'task_ratings' key

    Returns:
        Valid rating (0, 1, 2, 3, or "no_match") or None if no valid rating
    """
    if 'task_ratings' not in task:
        return None

    t1_rating = task['task_ratings'].get('t1')
    t2_rating = task['task_ratings'].get('t2')

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


def read_task_information(pickle_dir, view_type, ipsi_contra_csv=None):
    """
    Read task information and therapist labels from task pickle files.

    Task pickle structure (per entry):
        'patient_id': int
        'activity_id': int
        'CameraId': str (e.g., 'cam4')
        'frames': frame data
        'task_ratings': {'t1': rating, 't2': rating}

    Unlike segment pickles, task pickles have NO segment_id and use
    'task_ratings' instead of 'segment_ratings'.

    Args:
        pickle_dir: Directory containing task pickle files
        view_type: Camera view type ('top' or 'ipsi')
        ipsi_contra_csv: CSV mapping patient IDs to ipsilateral camera IDs

    Returns:
        Tuple of (train_tasks, inference_tasks)
        - train_tasks: List of tasks with consensus labels for training
        - inference_tasks: List of all tasks with individual t1/t2 labels
    """
    logger.info(f"Reading task information from {pickle_dir}")

    # Load patient to ipsilateral camera mapping
    patient_to_ipsilateral = {}
    if ipsi_contra_csv and os.path.exists(ipsi_contra_csv):
        camera_df = pd.read_csv(ipsi_contra_csv)
        patient_to_ipsilateral = dict(zip(camera_df['patient_id'], camera_df['ipsilateral_camera_id']))
        logger.info(f"Loaded ipsilateral camera mapping for {len(patient_to_ipsilateral)} patients")

    # Find all pickle files
    pickle_files = glob.glob(os.path.join(pickle_dir, "*.pkl"))
    logger.info(f"Found {len(pickle_files)} pickle files to process.")

    # Statistics counters
    rater_keys = ('t1', 't2')
    rating_values = (None, 0, 1, 2, 3)
    counts_task = {th: {val: 0 for val in rating_values} for th in rater_keys}
    seen = set()  # track (patient, activity, therapist) to avoid double-counting

    r0_count = 0
    r1_count = 0
    r2_count = 0
    r3_count = 0
    no_match_count = 0

    train_tasks = []
    inference_tasks = []

    for pkl_file in tqdm(pickle_files, desc="Reading task files"):
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading pickle file {pkl_file}: {e}")
            continue

        # Task pickles organized by camera_id, then groups, then individual tasks
        for camera_id in data:
            for task_group in data[camera_id]:
                # Handle both list-of-tasks and single-task structures
                tasks = task_group if isinstance(task_group, list) else [task_group]
                for task in tasks:
                    if not isinstance(task, dict):
                        continue

                    patient_id = task['patient_id']
                    task_camera_id = task['CameraId']
                    activity_id = task['activity_id']

                    # Filter by view_type
                    if view_type == 'top' and task_camera_id != 'cam3':
                        continue

                    if view_type == 'ipsi':
                        ipsilateral_camera = patient_to_ipsilateral.get(patient_id)
                        if ipsilateral_camera != task_camera_id:
                            continue

                    # Count task-level ratings (deduplicated per patient/activity/therapist)
                    tr = task.get('task_ratings', {})
                    for th in rater_keys:
                        key = (patient_id, activity_id, th)
                        if key in seen:
                            continue
                        seen.add(key)
                        rating = tr.get(th)
                        counts_task[th][rating] += 1

                    # Determine actual view type
                    actual_view_type = 'top' if task_camera_id == 'cam3' else 'ipsi'

                    # Extract individual ratings for inference
                    task_ratings = task.get('task_ratings', {})
                    t1_rating = task_ratings.get('t1')
                    t2_rating = task_ratings.get('t2')

                    t1_label = None if t1_rating is None else t1_rating
                    t2_label = None if t2_rating is None else t2_rating

                    video_id = f"patient_{patient_id}_task_{activity_id}_{task_camera_id}"

                    # Add to inference_tasks (all tasks)
                    inference_tasks.append({
                        'frames': task['frames'],
                        'video_id': video_id,
                        't1_label': t1_label,
                        't2_label': t2_label,
                        'camera_id': task_camera_id,
                        'view_type': actual_view_type,
                        'patient_id': patient_id,
                        'activity_id': activity_id,
                    })

                    # For training, need consensus rating from task_ratings
                    rating = get_valid_task_rating(task)

                    if rating is None or (rating != "no_match" and rating not in [0, 1, 2, 3]):
                        continue

                    if rating == "no_match":
                        no_match_count += 1
                        continue

                    try:
                        rating = int(rating)
                        label = rating
                        if rating == 0:
                            r0_count += 1
                        elif rating == 1:
                            r1_count += 1
                        elif rating == 2:
                            r2_count += 1
                        elif rating == 3:
                            r3_count += 1
                    except (ValueError, TypeError):
                        logger.warning(f"Skipping task with invalid rating: {rating}")
                        continue

                    train_tasks.append({
                        'frames': task['frames'],
                        'video_id': video_id,
                        'label': label,
                        'camera_id': task_camera_id,
                        'view_type': actual_view_type,
                        'patient_id': patient_id,
                        'activity_id': activity_id,
                    })

    # Log statistics
    logger.info(f"Task-level rating counts per therapist:")
    for th in rater_keys:
        logger.info(f"  {th}: {dict(counts_task[th])}")

    logger.info(f"Read {len(train_tasks)} tasks with valid ratings for training")
    logger.info(f"  Class 0 (rating 0): {r0_count}")
    logger.info(f"  Class 1 (rating 1): {r1_count}")
    logger.info(f"  Class 2 (rating 2): {r2_count}")
    logger.info(f"  Class 3 (rating 3): {r3_count}")
    logger.info(f"  No match: {no_match_count}")
    logger.info(f"Read {len(inference_tasks)} tasks for inference")

    return train_tasks, inference_tasks
