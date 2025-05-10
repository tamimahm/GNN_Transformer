import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch_geometric.data import Data
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def visualize_segment_graphs(segment_db_path, output_dir, view_type='top', num_samples=10, random_seed=42):
    """
    Visualize graphs from segment database before training
    
    Args:
        segment_db_path: Path to segment database
        output_dir: Directory to save visualizations
        view_type: Camera view type ('top' or 'ipsi')
        num_samples: Number of samples to visualize
        random_seed: Random seed for reproducibility
    """
    logger.info(f"Visualizing graphs from {segment_db_path}")
    
    # Set random seed
    np.random.seed(random_seed)
    
    # Create output directory
    vis_dir = os.path.join(output_dir, 'graph_visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Load segment database
    with open(segment_db_path, 'rb') as f:
        segment_data = pickle.load(f)
    
    logger.info(f"Loaded {len(segment_data)} segments")
    
    # Filter segments by view type
    view_segments = [
        (seg_id, segment) for seg_id, segment in segment_data.items()
        if segment['view_type'] == view_type
    ]
    
    logger.info(f"Found {len(view_segments)} segments for {view_type} view")
    
    # Select random samples
    if len(view_segments) > num_samples:
        sample_indices = np.random.choice(len(view_segments), num_samples, replace=False)
        selected_segments = [view_segments[i] for i in sample_indices]
    else:
        selected_segments = view_segments
    
    # Visualize each sample
    for i, (seg_id, segment) in enumerate(selected_segments):
        logger.info(f"Visualizing segment {i+1}/{len(selected_segments)}")
        
        # Extract data
        body_keypoints = segment['body_keypoints']
        hand_keypoints = segment.get('hand_keypoints')
        object_locations = segment.get('object_locations')
        label = segment.get('label', None)
        t1_label = segment.get('t1_label', None)
        t2_label = segment.get('t2_label', None)
        impaired_hand = segment.get('impaired_hand', 0)
        
        # Create figure
        visualize_keypoint_graph(
            body_keypoints, hand_keypoints, object_locations,
            vis_dir, i, seg_id, label, t1_label, t2_label, impaired_hand,
            frames_to_show=[0, 10, 19],  # Show first, middle, and last frame
            view_type=view_type
        )
    
    logger.info(f"Visualizations saved to {vis_dir}")
    return vis_dir

def visualize_keypoint_graph(body_keypoints, hand_keypoints, object_locations, 
                            output_dir, sample_index, segment_id, label=None, 
                            t1_label=None, t2_label=None, impaired_hand=0,
                            frames_to_show=None, view_type='top'):
    """
    Create visualization of keypoint graph with different colors for types
    
    Args:
        body_keypoints: Body keypoints array
        hand_keypoints: Hand keypoints array
        object_locations: Object locations array
        output_dir: Directory to save visualization
        sample_index: Index for naming
        segment_id: Segment ID
        label: Segment label
        t1_label: Therapist 1 label
        t2_label: Therapist 2 label
        impaired_hand: Which hand is impaired (0 for right, 1 for left)
        frames_to_show: Which frames to visualize (default: first, middle, last)
        view_type: View type ('top' or 'ipsi')
    """
    # Determine which frames to show
    if frames_to_show is None:
        # Get total frames
        num_frames = body_keypoints.shape[2]
        # Show first, middle and last frame
        frames_to_show = [0, num_frames // 2, num_frames - 1]
    
    # Create figure with subplots for selected frames
    fig, axes = plt.subplots(1, len(frames_to_show), figsize=(5 * len(frames_to_show), 5))
    
    # Handle case of single frame
    if len(frames_to_show) == 1:
        axes = [axes]
    
    # Set title with label information
    if label is not None:
        title = f"Sample {sample_index} (Label: {label})"
    elif t1_label is not None or t2_label is not None:
        title = f"Sample {sample_index} (T1: {t1_label}, T2: {t2_label})"
    else:
        title = f"Sample {sample_index}"
    
    plt.suptitle(f"{title} - {view_type.upper()} View")
    
    # Define names based on impaired side
    if impaired_hand == 0:  # Right impaired
        joint_names = ["Neck", "RShoulder", "RElbow", "RWrist", "MidHip"]
        body_key_indices = [1, 2, 3, 4, 8]  # Neck, RShoulder, RElbow, RWrist, MidHip
    else:  # Left impaired
        joint_names = ["Neck", "LShoulder", "LElbow", "LWrist", "MidHip"]
        body_key_indices = [1, 5, 6, 7, 8]  # Neck, LShoulder, LElbow, LWrist, MidHip
    
    # Define graph edges based on impaired side
    if impaired_hand == 0:  # Right impaired
        body_edges = [(0, 1), (1, 2), (2, 3), (0, 4)]  # Neck-RShoulder-RElbow-RWrist, Neck-MidHip
    else:  # Left impaired
        body_edges = [(0, 1), (1, 2), (2, 3), (0, 4)]  # Neck-LShoulder-LElbow-LWrist, Neck-MidHip
    
    # Process each selected frame
    for i, frame_idx in enumerate(frames_to_show):
        # Skip if frame index is beyond available frames
        if frame_idx >= body_keypoints.shape[2]:
            continue
        
        # Get keypoints for this frame
        body_frame = body_keypoints[:, :, frame_idx]
        
        # Extract the relevant keypoints
        if body_frame.shape[0] > max(body_key_indices):
            body_points = body_frame[body_key_indices]
        else:
            logger.warning(f"Not enough keypoints in body_frame: shape={body_frame.shape}")
            body_points = body_frame
        
        # # Get hand keypoints for this frame (if available)
        # hand_frame = None
        # if hand_keypoints is not None and frame_idx < hand_keypoints.shape[2]:
        #     hand_frame = hand_keypoints[:, :, frame_idx]
        
        # # Get object locations for this frame (if available)
        # object_frame = None
        # if object_locations is not None and frame_idx < object_locations.shape[2]:
        #     object_frame = object_locations[:, :, frame_idx]
        
        # Plot on the appropriate subplot
        ax = axes[i]
        ax.set_title(f"Frame {frame_idx}")
        
        # Plot body keypoints with labels
        if body_points is not None and body_points.shape[0] > 0:
            # Plot body joints
            ax.scatter(body_points[:, 0], body_points[:, 1], color='blue', s=30, label='Body Joints')
            
            # Add joint labels
            for j, name in enumerate(joint_names):
                if j < body_points.shape[0]:
                    ax.annotate(name, (body_points[j, 0], body_points[j, 1]), 
                                fontsize=8, ha='right', va='bottom')
            
            # Plot body connections
            for src, dst in body_edges:
                if src < body_points.shape[0] and dst < body_points.shape[0]:
                    ax.plot([body_points[src, 0], body_points[dst, 0]], 
                           [body_points[src, 1], body_points[dst, 1]], 'b-', alpha=0.6)
        
        # # Plot hand keypoints
        # if hand_frame is not None:
        #     # Determine fingertip indices based on hand keypoint format
        #     if hand_frame.shape[0] >= 21:  # MediaPipe format
        #         fingertips = [4, 8, 12, 16, 20]  # Indices for fingertips in MediaPipe
        #     else:
        #         # Adjust based on your actual hand keypoint format
        #         fingertips = list(range(min(5, hand_frame.shape[0])))
                
        #     finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
            
        #     # Plot points
        #     ax.scatter(hand_frame[:, 0], hand_frame[:, 1], color='green', s=20, label='Hand Joints')
            
            # # Add labels for fingertips
            # for j, tip_idx in enumerate(fingertips):
            #     if tip_idx < hand_frame.shape[0] and j < len(finger_names):
            #         ax.annotate(finger_names[j], (hand_frame[tip_idx, 0], hand_frame[tip_idx, 1]), 
            #                     fontsize=8, ha='left', va='top')
            
            # # Add label for wrist
            # if hand_frame.shape[0] > 0:
            #     ax.annotate("Hand Wrist", (hand_frame[0, 0], hand_frame[0, 1]), 
            #                 fontsize=8, ha='left', va='bottom')
            
        #     # Plot connections from wrist to fingertips
        #     for tip_idx in fingertips:
        #         if tip_idx < hand_frame.shape[0]:
        #             ax.plot([hand_frame[0, 0], hand_frame[tip_idx, 0]], 
        #                    [hand_frame[0, 1], hand_frame[tip_idx, 1]], 'g-', alpha=0.6)
        
        # # Plot object locations
        # if object_frame is not None and object_frame.shape[0] > 0:
        #     # Plot points
        #     ax.scatter(object_frame[:, 0], object_frame[:, 1], color='red', s=40, label='Objects')
            
            # # Add labels
            # for j in range(object_frame.shape[0]):
            #     ax.annotate(f"Obj {j+1}", (object_frame[j, 0], object_frame[j, 1]), 
            #                 fontsize=8, ha='center', va='bottom')
            
            # # Plot connections to body wrist if available
            # if body_points is not None and body_points.shape[0] >= 4:
            #     wrist_idx = 3  # Wrist is at index 3 in our filtered keypoints
            #     for j in range(object_frame.shape[0]):
            #         ax.plot([body_points[wrist_idx, 0], object_frame[j, 0]], 
            #                [body_points[wrist_idx, 1], object_frame[j, 1]], 'r-', alpha=0.4)
            
            # # Plot connections to hand fingertips if available
            # if hand_frame is not None and len(fingertips) > 0:
            #     for j in range(object_frame.shape[0]):
            #         for tip_idx in fingertips:
            #             if tip_idx < hand_frame.shape[0]:
            #                 ax.plot([hand_frame[tip_idx, 0], object_frame[j, 0]], 
            #                        [hand_frame[tip_idx, 1], object_frame[j, 1]], 'r-', alpha=0.2)
        
        # Add legend (only once)
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc='upper right')
        
        # Key adjustment: Keep the original OpenPose coordinate system
        # X increases from left to right, Y increases from top to bottom
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(1.05, -0.05)  # Reversed Y-axis to match OpenPose coordinates
        
        # Set equal aspect ratio
        ax.set_aspect('equal')
        
        # Add grid for better visibility
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add axes labels to clarify the coordinate system
        if i == 0:
            ax.set_xlabel("X (increases →)")
            ax.set_ylabel("Y (increases ↓)")
            
            # Add a note about coordinate system
            ax.text(0.02, 0.02, "OpenPose coordinates:\n(0,0) = top-left corner", 
                   transform=ax.transAxes, fontsize=8, 
                   bbox=dict(facecolor='white', alpha=0.7))
    
    # Save the visualization
    output_path = os.path.join(output_dir, f"graph_sample_{sample_index}_seg_{segment_id}_{view_type}.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    # Return the output path
    return output_path

# Usage example
if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Visualize graphs before training')
    parser.add_argument('--segment_db_path', type=str, required=True, help='Path to segment database')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save visualizations')
    parser.add_argument('--view_type', type=str, default='top', choices=['top', 'ipsi'], help='Camera view type')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to visualize')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Visualize graphs
    visualize_segment_graphs(
        args.segment_db_path,
        args.output_dir,
        args.view_type,
        args.num_samples,
        args.random_seed
    )