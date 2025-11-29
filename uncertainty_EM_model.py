import numpy as np
import os
import pandas as pd
def entropy(p, eps=1e-12):
    """
    Shannon entropy (nats) for a distribution p over classes, along last axis.
    p: [..., K]
    """
    p = np.clip(p, eps, 1 - eps)
    return -(p * np.log(p)).sum(axis=-1)
def ds_multiclass(R, K=4, max_iter=50, tol=1e-5, alpha=1e-2, verbose=False):
    """
    Dawid–Skene style EM for multiclass label aggregation.
    R: [N, C] int array with labels in 0..K-1 or -1 for missing.
    Returns:
      posteriors: [N, K]  p(T_i = k)
      pi:         [K]     class prior
      Theta:      [C, K, K] confusion matrices per clinician
    """
    R = np.asarray(R, dtype=int)
    N, C = R.shape
    mask_obs = R >= 0

    # ---- init pi from overall observed frequencies ----
    labels_flat = R[mask_obs]
    if labels_flat.size > 0:
        counts = np.bincount(labels_flat, minlength=K).astype(float)
        pi = counts / counts.sum()
    else:
        pi = np.ones(K) / K

    # ---- init Theta as near-identity for all raters ----
    Theta = np.zeros((C, K, K), dtype=float)
    for c in range(C):
        Theta[c] = np.eye(K) * 0.9 + (1 - np.eye(K)) * 0.1 / (K - 1)

    eps = 1e-12

    for it in range(max_iter):
        # ---------- E-step: posteriors over T_i ----------
        log_pi = np.log(pi + eps)           # [K]
        log_Theta = np.log(Theta + eps)     # [C, K, K]

        log_p = np.tile(log_pi, (N, 1))     # [N, K]
        for c in range(C):
            obs_idx = mask_obs[:, c]
            if not np.any(obs_idx):
                continue
            labels_c = R[obs_idx, c]                # [M]
            # log_Theta[c,:,labels_c] => [K, M]; transpose to [M, K]
            log_p[obs_idx, :] += log_Theta[c][:, labels_c].T

        # normalize to get p(T_i = k)
        max_log = log_p.max(axis=1, keepdims=True)
        log_p_norm = log_p - max_log
        p = np.exp(log_p_norm)
        p /= p.sum(axis=1, keepdims=True)          # [N, K]

        # ---------- M-step: update pi, Theta ----------
        pi_new = (p.sum(axis=0) + alpha) / (N + K * alpha)

        Theta_new = np.zeros_like(Theta)
        for c in range(C):
            num = np.zeros((K, K), dtype=float)
            denom = np.zeros(K, dtype=float)
            for i in range(N):
                if not mask_obs[i, c]:
                    continue
                l = R[i, c]               # observed label
                denom += p[i, :]          # contributes to all possible true labels
                num[:, l] += p[i, :]      # contributes to column l
            Theta_new[c] = (num + alpha) / (denom[:, None] + K * alpha)

        # convergence check
        delta_pi = np.max(np.abs(pi_new - pi))
        delta_theta = np.max(np.abs(Theta_new - Theta))
        if verbose:
            print(f"[task-only] Iter {it}: Δpi={delta_pi:.3e}, ΔTheta={delta_theta:.3e}")
        pi, Theta = pi_new, Theta_new
        if max(delta_pi, delta_theta) < tol:
            if verbose:
                print("[task-only] Converged.")
            break

    return p, pi, Theta

def em_task_segment(R_task, R_seg, K=4, max_iter=30, tol=1e-4, alpha=1e-2, verbose=False):
    """
    Hierarchical EM with latent task labels T_i and segment labels S_it.

    Generative model:
      T_i ~ Cat(pi)
      S_it | T_i=k ~ Cat(Phi_t[k, :])   for segments t=0..T-1
      R_task[i,c]    | T_i=k   ~ Cat(Theta_task[c, k, :])
      R_seg[i,t,c]   | S_it=s  ~ Cat(Theta_seg[c, s, :])

    Inputs:
      R_task: [N, C_task], labels in [0..K-1] or -1
      R_seg:  [N, T, C_seg], labels in [0..K-1] or -1

    Returns:
      gamma_T:    [N, K]       posterior over T_i
      q_S:        [N, T, K]    posterior over S_it
      pi:         [K]
      Theta_task: [C_task, K, K]
      Theta_seg:  [C_seg, K, K]
      Phi_t:      [T, K, K]    P(S_it = s | T_i = k) per segment index t
    """
    R_task = np.asarray(R_task, dtype=int)
    R_seg = np.asarray(R_seg, dtype=int)
    N, C_task = R_task.shape
    N2, T, C_seg = R_seg.shape
    assert N2 == N
    eps = 1e-12

    mask_task = R_task >= 0
    mask_seg = R_seg >= 0

    # ---- init pi from task labels ----
    labels_flat = R_task[mask_task]
    if labels_flat.size > 0:
        counts = np.bincount(labels_flat, minlength=K).astype(float)
        pi = counts / counts.sum()
    else:
        pi = np.ones(K) / K

    # ---- init Theta_task & Theta_seg as near-identity ----
    Theta_task = np.zeros((C_task, K, K))
    Theta_seg = np.zeros((C_seg, K, K))
    for c in range(C_task):
        Theta_task[c] = np.eye(K) * 0.9 + (1 - np.eye(K)) * 0.1 / (K - 1)
    for c in range(C_seg):
        Theta_seg[c] = np.eye(K) * 0.9 + (1 - np.eye(K)) * 0.1 / (K - 1)

    # ---- init Phi_t as near-identity (segment scores ≈ task scores) ----
    Phi_t = np.zeros((T, K, K))
    for t_idx in range(T):
        Phi_t[t_idx] = np.eye(K) * 0.9 + (1 - np.eye(K)) * 0.1 / (K - 1)

    for it in range(max_iter):
        # ---------- E-step ----------
        log_pi = np.log(pi + eps)
        log_Theta_task = np.log(Theta_task + eps)  # [C_task, K, K]
        log_Theta_seg = np.log(Theta_seg + eps)    # [C_seg, K, K]
        log_Phi_t = np.log(Phi_t + eps)            # [T, K, K]

        # log p(R_task_i | T_i=k)
        log_p_Rtask = np.zeros((N, K))
        for c in range(C_task):
            obs_idx = mask_task[:, c]
            if not np.any(obs_idx):
                continue
            labels_c = R_task[obs_idx, c]
            log_p_Rtask[obs_idx, :] += log_Theta_task[c][:, labels_c].T

        # log p(R_seg_it | S_it=s)
        log_p_Rseg = np.zeros((N, T, K))
        for c in range(C_seg):
            obs_idx_it = mask_seg[:, :, c]   # [N, T]
            idx_pairs = np.argwhere(obs_idx_it)
            if idx_pairs.size == 0:
                continue
            i_idx = idx_pairs[:, 0]
            t_idx = idx_pairs[:, 1]
            labels = R_seg[i_idx, t_idx, c]
            log_theta_c = log_Theta_seg[c]   # [K, K]
            contrib = log_theta_c[:, labels].T  # [M, K]
            for row, (ii, tt) in enumerate(zip(i_idx, t_idx)):
                log_p_Rseg[ii, tt, :] += contrib[row]

        # log_A_it[i,t,k] = log sum_s Phi_t[t,k,s] * p(R_seg_it | S_it=s)
        log_A_it = np.zeros((N, T, K))
        for t_idx in range(T):
            lp_t = log_Phi_t[t_idx]  # [K, K]
            for k in range(K):
                temp = lp_t[k, :] + log_p_Rseg[:, t_idx, :]  # [N, K] over s
                m = temp.max(axis=1, keepdims=True)
                log_A_it[:, t_idx, k] = (m.squeeze() +
                                         np.log(np.exp(temp - m).sum(axis=1) + eps))

        # sum over segments
        sum_log_A = log_A_it.sum(axis=1)  # [N, K]

        # log posterior over T (unnormalized), then normalize -> gamma_T
        log_p_T_unnorm = log_pi[None, :] + log_p_Rtask + sum_log_A
        mT = log_p_T_unnorm.max(axis=1, keepdims=True)
        log_p_T_norm = log_p_T_unnorm - mT
        gamma_T = np.exp(log_p_T_norm)
        gamma_T /= gamma_T.sum(axis=1, keepdims=True)  # [N, K]

        # joint for T, S_it: xi[i,k,t,s]
        xi = np.zeros((N, K, T, K))
        for t_idx in range(T):
            # sum_{t' != t} log_A_it
            log_A_excl_t = sum_log_A - log_A_it[:, t_idx, :]  # [N, K]
            lp_t = log_Phi_t[t_idx]                           # [K, K]
            for k in range(K):
                base = log_pi[k] + log_p_Rtask[:, k] + log_A_excl_t[:, k]  # [N]
                for s in range(K):
                    xi_log = base + lp_t[k, s] + log_p_Rseg[:, t_idx, s]   # [N]
                    # Normalization constant log_Z_i is the log p(data_i),
                    # which equals mT[i] + log sum_k exp(log_p_T_unnorm[i,k]-mT[i]).
                    # But since gamma_T is normalized, log_Z_i = mT.squeeze().
                    # So p(T=k,S_it=s | data) = exp(xi_log - log_Z_i).
                    xi[:, k, t_idx, s] = xi_log

        log_Z_i = mT.squeeze()  # [N]

        for t_idx in range(T):
            for k in range(K):
                for s in range(K):
                    xi[:, k, t_idx, s] = np.exp(xi[:, k, t_idx, s] - log_Z_i)

        # q_S[i,t,s] = sum_k xi[i,k,t,s]
        q_S = xi.sum(axis=1)  # [N, T, K]

        # ---------- M-step ----------
        # pi
        pi_new = (gamma_T.sum(axis=0) + alpha) / (N + K * alpha)

        # Theta_task
        Theta_task_new = np.zeros_like(Theta_task)
        for c in range(C_task):
            num = np.zeros((K, K))
            denom = np.zeros(K)
            for i in range(N):
                if not mask_task[i, c]:
                    continue
                l = R_task[i, c]
                denom += gamma_T[i, :]
                num[:, l] += gamma_T[i, :]
            Theta_task_new[c] = (num + alpha) / (denom[:, None] + K * alpha)

        # Theta_seg (using q_S)
        Theta_seg_new = np.zeros_like(Theta_seg)
        for c in range(C_seg):
            num = np.zeros((K, K))
            denom = np.zeros(K)
            for i in range(N):
                for t_idx in range(T):
                    if not mask_seg[i, t_idx, c]:
                        continue
                    l = R_seg[i, t_idx, c]
                    denom += q_S[i, t_idx, :]
                    num[:, l] += q_S[i, t_idx, :]
            Theta_seg_new[c] = (num + alpha) / (denom[:, None] + K * alpha)

        # Phi_t from xi (joint T,S per segment index)
        Phi_t_new = np.zeros_like(Phi_t)
        for t_idx in range(T):
            counts_TS = xi[:, :, t_idx, :].sum(axis=0)  # [K, K]
            Phi_t_new[t_idx] = ((counts_TS + alpha) /
                                (counts_TS.sum(axis=1, keepdims=True) + K * alpha))

        # convergence
        delta_pi = np.max(np.abs(pi_new - pi))
        delta_theta_task = np.max(np.abs(Theta_task_new - Theta_task))
        delta_theta_seg = np.max(np.abs(Theta_seg_new - Theta_seg))
        delta_phi = np.max(np.abs(Phi_t_new - Phi_t))
        if verbose:
            print(f"[task+seg] Iter {it}: Δpi={delta_pi:.3e}, "
                  f"ΔTheta_task={delta_theta_task:.3e}, "
                  f"ΔTheta_seg={delta_theta_seg:.3e}, "
                  f"ΔPhi={delta_phi:.3e}")
        pi, Theta_task, Theta_seg, Phi_t = pi_new, Theta_task_new, Theta_seg_new, Phi_t_new
        if max(delta_pi, delta_theta_task, delta_theta_seg, delta_phi) < tol:
            if verbose:
                print("[task+seg] Converged.")
            break

    return gamma_T, q_S, pi, Theta_task, Theta_seg, Phi_t


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
            print(f"Filename {file_name} does not match expected format. Skipping.")
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

# def load_rating_info(csv_dir):
#     """
#     Loads the task_final and segment_final CSV files containing ratings.
    
#     Expected files in csv_dir:
#       - task_final.csv with columns: PatientTaskHandMappingId, Completed, Initialized, Time, Impaired, Rating, TherapistId, CreatedAt, ModifiedAt, Finger
#       - segment_final.csv with columns: PatientTaskHandMappingId, SegmentId, Completed, Initialized, Time, Impaired, Rating, TherapistId, CreatedAt, ModifiedAt, Finger
      
#     Returns:
#       Two dictionaries:
#          task_ratings: mapping PatientTaskHandMappingId to a dictionary with keys 't1' and (optionally) 't2'
#                        for task ratings.
#          segment_ratings: mapping PatientTaskHandMappingId to a dictionary with keys 't1' and (optionally) 't2',
#                           where each value is itself a dictionary mapping SegmentId to its rating.
#     """   
#     # task_file = os.path.join(csv_dir, "task_final_updated.csv")
#     # segment_file = os.path.join(csv_dir, "segment_final_updated.csv")
#     task_file = os.path.join(csv_dir, "task_final.csv")
#     segment_file = os.path.join(csv_dir, "segment_final.csv")    
#     task_df = pd.read_csv(task_file)
#     segment_df = pd.read_csv(segment_file)
    
#     # Process task ratings: store first rating as 't1' and second (if available) as 't2'.
#     task_ratings = {}
#     for _, row in task_df.iterrows():
#         mapping_id = row['PatientTaskHandMappingId']
#         rating = row['Rating']
#         if pd.notna(rating):
#             if mapping_id not in task_ratings:
#                 task_ratings[mapping_id] = {'t1': rating}
#             elif 't2' not in task_ratings[mapping_id]:
#                 task_ratings[mapping_id]['t2'] = rating
#             # Ignore any additional ratings.
    
#     # Process segment ratings:
#     # For each mapping id and therapist, build a dictionary mapping each SegmentId to its rating.
#     segment_ratings = {}
#     grouped = segment_df.groupby(['PatientTaskHandMappingId', 'TherapistId'])
#     for (mapping_id, therapist_id), group in grouped:
#         seg_rating_dict = {}
#         for _, row in group.iterrows():
#             # Assuming segment_final.csv has a 'SegmentId' column.
#             seg_id = row['SegmentId']
#             rating = row['Rating']
#             if pd.notna(rating):
#                 seg_rating_dict[seg_id] = rating
#         if not seg_rating_dict:
#             continue
#         # For each mapping_id, store the first therapist's segment ratings as 't1'
#         # and if a second therapist is available, store their ratings as 't2'.
#         if mapping_id not in segment_ratings:
#             segment_ratings[mapping_id] = {'t1': seg_rating_dict}
#         elif 't1' in segment_ratings[mapping_id] and 't2' not in segment_ratings[mapping_id]:
#             segment_ratings[mapping_id]['t2'] = seg_rating_dict
#         # If already both t1 and t2 exist, ignore extra groups.
    
#     return task_ratings, segment_ratings


def load_rating_info(csv_dir):
    """
    Reads task_final.csv and segment_final.csv and returns:
        task_ratings       {mapping_id: {'t1': r, 't2': r2}}
        segment_ratings    {mapping_id: {'t1': {seg: r, ...}, 't2': {...}}}
        composite_ratings  {mapping_id: {'t1': {seg: '0,1,...'}, 't2': {...}}}
    The lower TherapistId per mapping is always “t1”, higher (if present) is “t2”.
    """
    task_df    = pd.read_csv(os.path.join(csv_dir, "task_final.csv"))
    seg_df     = pd.read_csv(os.path.join(csv_dir, "segment_final.csv"))

    # ------------------------------------------------------------------ #
    # 1) determine per‑mapping therapist roles (t1 / t2)                 #
    # ------------------------------------------------------------------ #
    therapist_map_per_mapping = {}

    def _assign_roles(group):
        tids = sorted(group["TherapistId"].dropna().unique())
        if tids:
            therapist_map_per_mapping[group.name] = {tids[0]: "t1"}
            if len(tids) > 1:
                therapist_map_per_mapping[group.name][tids[1]] = "t2"

    task_df.groupby("PatientTaskHandMappingId").apply(_assign_roles)
    seg_df.groupby("PatientTaskHandMappingId").apply(_assign_roles)

    # ------------------------------------------------------------------ #
    # 2) TASK‑level ratings                                              #
    # ------------------------------------------------------------------ #
    task_ratings = {}
    for _, row in task_df.iterrows():
        mid   = row["PatientTaskHandMappingId"]
        rid   = row["Rating"]
        tid   = row["TherapistId"]
        if pd.isna(rid) or pd.isna(tid):          # skip incomplete rows
            continue
        role = therapist_map_per_mapping.get(mid, {}).get(tid)
        if role is None:
            continue
        task_ratings.setdefault(mid, {})[role] = rid

    # ------------------------------------------------------------------ #
    # 3) SEGMENT & COMPOSITE ratings                                      #
    # ------------------------------------------------------------------ #
    feature_cols = [
        "SEAFR","TS","ROME","FPS","WPAT","HA","DP",
        "DPO","SAT","DMR","THS","PP","FPO","SA"
    ]

    segment_ratings   = {}
    composite_ratings = {}

    for _, row in seg_df.iterrows():
        mid = row["PatientTaskHandMappingId"]
        sid = row["SegmentId"]
        rid = row["Rating"]
        tid = row["TherapistId"]
        role = therapist_map_per_mapping.get(mid, {}).get(tid)
        if role is None:
            continue

        # -------- segment rating --------
        if pd.notna(rid):
            segment_ratings.setdefault(mid, {}).setdefault(role, {})[sid] = rid

        # -------- composite features ----
        bin_feats = []
        for col in feature_cols:
            v = row.get(col)
            bin_feats.append("1" if v == 1 else "0")
        comp_str = ",".join(bin_feats)
        composite_ratings.setdefault(mid, {}).setdefault(role, {})[sid] = comp_str

    return task_ratings, segment_ratings, composite_ratings
import numpy as np

def infer_mqe_dim(mqe_dict):
    """Infer number of MQE dimensions from the first non-empty MQE string."""
    for entry in mqe_dict.values():
        for ck in ("t1", "t2"):
            seg_map = entry.get(ck, {})
            for _, mqe_str in seg_map.items():
                parts = [p.strip() for p in mqe_str.split(",") if p.strip() != ""]
                return len(parts)
    raise ValueError("Could not infer MQE dimension from mqe_dict.")

def build_mqe_arrays(mqe_dict, idx_to_instance_id, T_seg=4, clinician_keys=("t1", "t2"), M_mqe=None):
    """
    Build MQE arrays aligned with idx_to_instance_id.

    Args:
      mqe_dict: dict[instance_id] -> {'t1': {seg: '0,1,...'}, 't2': {...}}
      idx_to_instance_id: list of instance ids in same order as R_task / R_seg rows.
      T_seg: number of segments (we assume 4).
      clinician_keys: names of clinician keys in dict.
      M_mqe: number of MQE dims; if None, infer from mqe_dict.

    Returns:
      X_mqe_task: [N, M_mqe] binary (0/1) per instance.
      X_mqe_seg:  [N, T_seg, M_mqe] binary per instance, per segment idx 0..3.
    """
    if M_mqe is None:
        M_mqe = infer_mqe_dim(mqe_dict)

    N = len(idx_to_instance_id)
    X_mqe_task = np.zeros((N, M_mqe), dtype=int)
    X_mqe_seg = np.zeros((N, T_seg, M_mqe), dtype=int)

    valid_segments = {1, 2, 3, 4}

    for i, inst_id in enumerate(idx_to_instance_id):
        entry = mqe_dict.get(inst_id, None)
        if entry is None:
            continue

        for ck in clinician_keys:
            seg_map = entry.get(ck, {})
            for seg_id, mqe_str in seg_map.items():
                seg_int = int(seg_id)
                if seg_int not in valid_segments:
                    # ignore weird segment numbers (7, 9, ...)
                    continue
                t_idx = seg_int - 1  # segment 1->0, 2->1, etc.

                parts = [p.strip() for p in mqe_str.split(",") if p.strip() != ""]
                if len(parts) != M_mqe:
                    raise ValueError(f"MQE length mismatch for instance {inst_id}, segment {seg_id}")
                vec = np.array(parts, dtype=int)

                # OR across clinicians / segments
                X_mqe_task[i, :] |= vec
                X_mqe_seg[i, t_idx, :] |= vec

    return X_mqe_task, X_mqe_seg
def em_task_mqe(R_task, X_mqe_task, K=4, max_iter=50, tol=1e-5,
                alpha=1e-2, alpha_beta=1e-2, verbose=False):
    """
    EM for latent task labels T with task ratings + MQEs as features.

    Inputs:
      R_task:      [N, C_task] int in 0..K-1 or -1
      X_mqe_task:  [N, M] binary (0/1) MQE indicators aggregated per task

    Returns:
      gamma_T:     [N, K]  posterior p(T_i=k | R_task, X_mqe_task)
      pi:          [K]
      Theta_task:  [C_task, K, K]
      Beta_task:   [K, M]  Bernoulli params per class & MQE
    """
    R_task = np.asarray(R_task, dtype=int)
    X_mqe_task = np.asarray(X_mqe_task, dtype=int)
    N, C_task = R_task.shape
    N2, M = X_mqe_task.shape
    assert N2 == N
    eps = 1e-12

    mask_task = R_task >= 0

    # init pi from task labels
    labels_flat = R_task[mask_task]
    if labels_flat.size > 0:
        counts = np.bincount(labels_flat, minlength=K).astype(float)
        pi = counts / counts.sum()
    else:
        pi = np.ones(K) / K

    # init Theta_task as near-identity
    Theta_task = np.zeros((C_task, K, K))
    for c in range(C_task):
        Theta_task[c] = np.eye(K) * 0.9 + (1 - np.eye(K)) * 0.1 / (K - 1)

    # init Beta_task from global MQE frequencies (same for all classes at start)
    freq_mqe = X_mqe_task.mean(axis=0)  # [M]
    # avoid 0 or 1
    freq_mqe = np.clip(freq_mqe, 0.05, 0.95)
    Beta_task = np.tile(freq_mqe[None, :], (K, 1))  # [K, M]

    for it in range(max_iter):
        # ---------- E-step ----------
        log_pi = np.log(pi + eps)
        log_Theta_task = np.log(Theta_task + eps)  # [C_task, K, K]

        # log p(R_task | T_i=k)
        log_p_Rtask = np.zeros((N, K))
        for c in range(C_task):
            obs_idx = mask_task[:, c]
            if not np.any(obs_idx):
                continue
            labels_c = R_task[obs_idx, c]
            log_p_Rtask[obs_idx, :] += log_Theta_task[c][:, labels_c].T

        # log p(X_mqe_task | T_i=k)
        log_Beta = np.log(Beta_task + eps)         # [K, M]
        log_1mB  = np.log(1 - Beta_task + eps)     # [K, M]
        # X is [N, M]; we want [N, K]
        # log p(x | k) = sum_m x*log_Beta[k,m] + (1-x)*log(1-Beta[k,m])
        log_p_MQE = np.zeros((N, K))
        for k in range(K):
            log_p_MQE[:, k] = (X_mqe_task * log_Beta[k, :]).sum(axis=1) + \
                              ((1 - X_mqe_task) * log_1mB[k, :]).sum(axis=1)

        # log posterior up to constant
        log_p = log_pi[None, :] + log_p_Rtask + log_p_MQE
        max_log = log_p.max(axis=1, keepdims=True)
        log_p_norm = log_p - max_log
        gamma_T = np.exp(log_p_norm)
        gamma_T /= gamma_T.sum(axis=1, keepdims=True)  # [N, K]

        # ---------- M-step ----------
        # pi
        pi_new = (gamma_T.sum(axis=0) + alpha) / (N + K * alpha)

        # Theta_task
        Theta_task_new = np.zeros_like(Theta_task)
        for c in range(C_task):
            num = np.zeros((K, K))
            denom = np.zeros(K)
            for i in range(N):
                if not mask_task[i, c]:
                    continue
                l = R_task[i, c]
                denom += gamma_T[i, :]
                num[:, l] += gamma_T[i, :]
            Theta_task_new[c] = (num + alpha) / (denom[:, None] + K * alpha)

        # Beta_task
        Beta_task_new = np.zeros_like(Beta_task)
        for k in range(K):
            gamma_k = gamma_T[:, k]  # [N]
            denom_k = gamma_k.sum()
            if denom_k <= 0:
                Beta_task_new[k, :] = Beta_task[k, :]
            else:
                num_k = (gamma_k[:, None] * X_mqe_task).sum(axis=0)  # [M]
                Beta_task_new[k, :] = (num_k + alpha_beta) / (denom_k + 2 * alpha_beta)

        # convergence
        delta_pi = np.max(np.abs(pi_new - pi))
        delta_theta = np.max(np.abs(Theta_task_new - Theta_task))
        delta_beta = np.max(np.abs(Beta_task_new - Beta_task))
        if verbose:
            print(f"[task+MQE] Iter {it}: Δpi={delta_pi:.3e}, "
                  f"ΔTheta={delta_theta:.3e}, ΔBeta={delta_beta:.3e}")
        pi, Theta_task, Beta_task = pi_new, Theta_task_new, Beta_task_new
        if max(delta_pi, delta_theta, delta_beta) < tol:
            if verbose:
                print("[task+MQE] Converged.")
            break

    return gamma_T, pi, Theta_task, Beta_task
def em_task_seg_mqe(R_task, R_seg, X_mqe_seg,
                    K=4, max_iter=50, tol=1e-5,
                    alpha=1e-2, alpha_beta=1e-2,
                    verbose=False):
    """
    EM for latent task labels T with:
      - task ratings R_task
      - segment ratings R_seg
      - segment MQEs X_mqe_seg

    Inputs:
      R_task:    [N, C_task] int in 0..K-1 or -1
      R_seg:     [N, T_seg, C_seg] int in 0..K-1 or -1
      X_mqe_seg: [N, T_seg, M] binary (0/1)

    Returns:
      gamma_T:      [N, K]  posterior p(T_i=k | all)
      pi:           [K]
      Theta_task:   [C_task, K, K]
      Theta_seg:    [C_seg, K, K]
      Beta_seg:     [K, M]
    """
    R_task = np.asarray(R_task, dtype=int)
    R_seg = np.asarray(R_seg, dtype=int)
    X_mqe_seg = np.asarray(X_mqe_seg, dtype=int)

    N, C_task = R_task.shape
    N2, T_seg, C_seg = R_seg.shape
    N3, T_seg2, M = X_mqe_seg.shape
    assert N2 == N and N3 == N and T_seg2 == T_seg

    eps = 1e-12
    mask_task = R_task >= 0
    mask_seg = R_seg >= 0

    # ---- init pi from task labels ----
    labels_flat = R_task[mask_task]
    if labels_flat.size > 0:
        counts = np.bincount(labels_flat, minlength=K).astype(float)
        pi = counts / counts.sum()
    else:
        pi = np.ones(K) / K

    # ---- init Theta_task & Theta_seg near-identity ----
    Theta_task = np.zeros((C_task, K, K))
    Theta_seg = np.zeros((C_seg, K, K))
    for c in range(C_task):
        Theta_task[c] = np.eye(K) * 0.9 + (1 - np.eye(K)) * 0.1 / (K - 1)
    for c in range(C_seg):
        Theta_seg[c] = np.eye(K) * 0.9 + (1 - np.eye(K)) * 0.1 / (K - 1)

    # ---- init Beta_seg from global MQE frequencies ----
    freq_mqe_seg = X_mqe_seg.reshape(N * T_seg, M).mean(axis=0)
    freq_mqe_seg = np.clip(freq_mqe_seg, 0.05, 0.95)
    Beta_seg = np.tile(freq_mqe_seg[None, :], (K, 1))  # [K, M]

    for it in range(max_iter):
        # ---------- E-step ----------
        log_pi = np.log(pi + eps)
        log_Theta_task = np.log(Theta_task + eps)  # [C_task, K, K]
        log_Theta_seg = np.log(Theta_seg + eps)    # [C_seg, K, K]
        log_Beta = np.log(Beta_seg + eps)         # [K, M]
        log_1mB  = np.log(1 - Beta_seg + eps)     # [K, M]

        # log p(R_task | T_i=k)
        log_p_Rtask = np.zeros((N, K))
        for c in range(C_task):
            obs_idx = mask_task[:, c]
            if not np.any(obs_idx):
                continue
            labels_c = R_task[obs_idx, c]
            log_p_Rtask[obs_idx, :] += log_Theta_task[c][:, labels_c].T

        # log p(R_seg | T_i=k)
        log_p_Rseg = np.zeros((N, K))
        for c in range(C_seg):
            obs_idx_it = mask_seg[:, :, c]  # [N, T_seg]
            idx_pairs = np.argwhere(obs_idx_it)
            if idx_pairs.size == 0:
                continue
            i_idx = idx_pairs[:, 0]
            t_idx = idx_pairs[:, 1]
            labels = R_seg[i_idx, t_idx, c]  # [M_pairs]
            log_theta_c = log_Theta_seg[c]   # [K, K]
            contrib = log_theta_c[:, labels].T  # [M_pairs, K]
            for row, ii in enumerate(i_idx):
                log_p_Rseg[ii, :] += contrib[row]

        # log p(X_mqe_seg | T_i=k)
        # X_mqe_seg: [N, T_seg, M]
        log_p_MQE_seg = np.zeros((N, K))
        for k in range(K):
            # for each i,t: sum_m x_itm*log_Beta[k,m] + (1-x)*log(1-Beta[k,m])
            # flatten (N,T_seg) for vectorization
            X_flat = X_mqe_seg.reshape(N * T_seg, M)  # [N*T_seg, M]
            lp_flat = (X_flat * log_Beta[k, :]).sum(axis=1) + \
                      ((1 - X_flat) * log_1mB[k, :]).sum(axis=1)  # [N*T_seg]
            lp = lp_flat.reshape(N, T_seg).sum(axis=1)  # sum over segments -> [N]
            log_p_MQE_seg[:, k] = lp

        # combine all
        log_p = log_pi[None, :] + log_p_Rtask + log_p_Rseg + log_p_MQE_seg
        max_log = log_p.max(axis=1, keepdims=True)
        log_p_norm = log_p - max_log
        gamma_T = np.exp(log_p_norm)
        gamma_T /= gamma_T.sum(axis=1, keepdims=True)  # [N, K]

        # ---------- M-step ----------
        # pi
        pi_new = (gamma_T.sum(axis=0) + alpha) / (N + K * alpha)

        # Theta_task
        Theta_task_new = np.zeros_like(Theta_task)
        for c in range(C_task):
            num = np.zeros((K, K))
            denom = np.zeros(K)
            for i in range(N):
                if not mask_task[i, c]:
                    continue
                l = R_task[i, c]
                denom += gamma_T[i, :]
                num[:, l] += gamma_T[i, :]
            Theta_task_new[c] = (num + alpha) / (denom[:, None] + K * alpha)

        # Theta_seg
        Theta_seg_new = np.zeros_like(Theta_seg)
        for c in range(C_seg):
            num = np.zeros((K, K))
            denom = np.zeros(K)
            for i in range(N):
                for t_idx in range(T_seg):
                    if not mask_seg[i, t_idx, c]:
                        continue
                    l = R_seg[i, t_idx, c]
                    denom += gamma_T[i, :]
                    num[:, l] += gamma_T[i, :]
            Theta_seg_new[c] = (num + alpha) / (denom[:, None] + K * alpha)

        # Beta_seg
        Beta_seg_new = np.zeros_like(Beta_seg)
        for k in range(K):
            gamma_k = gamma_T[:, k]  # [N]
            # each instance contributes T_seg times
            denom_k = (gamma_k.sum() * T_seg)
            if denom_k <= 0:
                Beta_seg_new[k, :] = Beta_seg[k, :]
            else:
                # sum over i,t
                num_k = 0.0
                for i in range(N):
                    num_k += gamma_k[i] * X_mqe_seg[i, :, :].sum(axis=0)  # [M]
                Beta_seg_new[k, :] = (num_k + alpha_beta) / (denom_k + 2 * alpha_beta)

        # convergence
        delta_pi = np.max(np.abs(pi_new - pi))
        delta_theta_task = np.max(np.abs(Theta_task_new - Theta_task))
        delta_theta_seg = np.max(np.abs(Theta_seg_new - Theta_seg))
        delta_beta = np.max(np.abs(Beta_seg_new - Beta_seg))
        if verbose:
            print(f"[task+seg+MQE] Iter {it}: Δpi={delta_pi:.3e}, "
                  f"ΔTheta_task={delta_theta_task:.3e}, "
                  f"ΔTheta_seg={delta_theta_seg:.3e}, "
                  f"ΔBeta={delta_beta:.3e}")
        pi, Theta_task, Theta_seg, Beta_seg = pi_new, Theta_task_new, Theta_seg_new, Beta_seg_new
        if max(delta_pi, delta_theta_task, delta_theta_seg, delta_beta) < tol:
            if verbose:
                print("[task+seg+MQE] Converged.")
            break

    return gamma_T, pi, Theta_task, Theta_seg, Beta_seg


import numpy as np

def infer_mqe_dim_from_dict(mqe_dict):
    """Infer M_mqe (#MQE types) from first non-empty string."""
    for entry in mqe_dict.values():
        for ck in ("t1", "t2"):
            seg_map = entry.get(ck, {})
            for _, mqe_str in seg_map.items():
                parts = [p.strip() for p in mqe_str.split(",") if p.strip() != ""]
                return len(parts)
    raise ValueError("Could not infer MQE dimension from mqe_dict.")

def build_mqe_rater_arrays(mqe_dict, idx_to_instance_id,
                           T_seg=4, clinician_keys=("t1", "t2"), M_mqe=None):
    """
    Build Y_mqe[i, t, c, m] from your mqe_dict.

    Args:
      mqe_dict: dict[instance_id] -> {
          't1': {seg_id: '0,1,...'},
          't2': {seg_id: '0,1,...'},
      }
      idx_to_instance_id: list of instance ids in same order as R_task/R_seg rows.
      T_seg: number of segments (we assume 4).
      clinician_keys: ordered list of clinician keys to map to c dimension.
      M_mqe: number of MQE dims; if None, infer from dictionary.

    Returns:
      Y_mqe: [N, T_seg, C_mqe, M_mqe] with values in {0,1,-1},
             where -1 means missing (no annotation).
    """
    if M_mqe is None:
        M_mqe = infer_mqe_dim_from_dict(mqe_dict)

    N = len(idx_to_instance_id)
    C_mqe = len(clinician_keys)

    Y_mqe = np.full((N, T_seg, C_mqe, M_mqe), -1, dtype=int)
    valid_segments = set(range(1, T_seg + 1))

    for i, inst_id in enumerate(idx_to_instance_id):
        entry = mqe_dict.get(inst_id, None)
        if entry is None:
            continue

        for c, ck in enumerate(clinician_keys):
            seg_map = entry.get(ck, {})
            for seg_id, mqe_str in seg_map.items():
                seg_int = int(seg_id)
                if seg_int not in valid_segments:
                    # ignore weird segments like 7, 9
                    continue
                t_idx = seg_int - 1  # map 1..T_seg -> 0..T_seg-1

                parts = [p.strip() for p in mqe_str.split(",") if p.strip() != ""]
                if len(parts) != M_mqe:
                    raise ValueError(f"MQE length mismatch for instance {inst_id}, seg {seg_id}")
                vec = np.array(parts, dtype=int)

                Y_mqe[i, t_idx, c, :] = vec

    return Y_mqe

import numpy as np
import itertools

def em_full_TS_MQE(R_task, R_seg, Y_mqe,
                   K=4, max_iter=30, tol=1e-4,
                   alpha=1e-2, alpha_beta=1e-2,
                   verbose=False):
    """
    Full hierarchical EM for latent task labels T and segment labels S_it with
    task ratings, segment ratings, and clinician-specific MQEs.

    Generative assumptions:
      T_i ~ Cat(pi)
      S_it | T_i=k ~ Cat(Phi_t[k, :])   (t=0..T_seg-1)
      R_task[i,c]      | T_i=k ~ Cat(Theta_task[c, k, :])
      R_seg[i,t,c]     | S_it=s ~ Cat(Theta_seg[c, s, :])
      Y_mqe[i,t,c,m]   | S_it=s ~ Bernoulli(Beta_mqe[c, s, m])

    Inputs:
      R_task: [N, C_task] int in 0..K-1 or -1
      R_seg:  [N, T_seg, C_seg] int in 0..K-1 or -1
      Y_mqe:  [N, T_seg, C_mqe, M_mqe] int in {0,1,-1}

    Returns:
      gamma_T:    [N, K]         posterior p(T_i=k | all data)
      tau_S:      [N, T_seg, K]  posterior p(S_it=s | all data)
      pi:         [K]
      Theta_task: [C_task, K, K]
      Theta_seg:  [C_seg, K, K]
      Phi_t:      [T_seg, K, K]  P(S_it=s | T_i=k, segment_index=t)
      Beta_mqe:   [C_mqe, K, M_mqe]  P(Y=1 | S_it=s, clinician c, MQE m)
    """
    R_task = np.asarray(R_task, dtype=int)
    R_seg = np.asarray(R_seg, dtype=int)
    Y_mqe = np.asarray(Y_mqe, dtype=int)

    N, C_task = R_task.shape
    N2, T_seg, C_seg = R_seg.shape
    N3, T_seg2, C_mqe, M_mqe = Y_mqe.shape
    assert N2 == N and N3 == N and T_seg2 == T_seg

    eps = 1e-12
    mask_task = R_task >= 0
    mask_seg = R_seg >= 0
    mask_mqe = Y_mqe >= 0  # True where 0 or 1

    # ---- init pi from task labels (if available) ----
    labels_flat = R_task[mask_task]
    if labels_flat.size > 0:
        counts = np.bincount(labels_flat, minlength=K).astype(float)
        pi = counts / counts.sum()
    else:
        pi = np.ones(K) / K

    # ---- init Theta_task & Theta_seg as near-identity ----
    Theta_task = np.zeros((C_task, K, K))
    Theta_seg = np.zeros((C_seg, K, K))
    for c in range(C_task):
        Theta_task[c] = np.eye(K) * 0.9 + (1 - np.eye(K)) * 0.1 / (K - 1)
    for c in range(C_seg):
        Theta_seg[c] = np.eye(K) * 0.9 + (1 - np.eye(K)) * 0.1 / (K - 1)

    # ---- init Phi_t as near-identity (segment scores ≈ task scores) ----
    Phi_t = np.zeros((T_seg, K, K))
    for t_idx in range(T_seg):
        Phi_t[t_idx] = np.eye(K) * 0.9 + (1 - np.eye(K)) * 0.1 / (K - 1)

    # ---- init Beta_mqe from global MQE frequencies (ignoring S for init) ----
    # use only observed Y's, per clinician c and MQE m
    freq_mqe = np.zeros((C_mqe, M_mqe))
    for c in range(C_mqe):
        # Y_c: [N, T_seg, M_mqe], mask_c: [N, T_seg, M_mqe]
        Y_c = Y_mqe[:, :, c, :]
        mask_c = mask_mqe[:, :, c, :]
        # flatten (N, T_seg) -> N*T_seg for convenience
        Y_c_flat = Y_c.reshape(-1, M_mqe)         # [N*T_seg, M_mqe]
        mask_c_flat = mask_c.reshape(-1, M_mqe)   # [N*T_seg, M_mqe]
        for m in range(M_mqe):
            mask_m = mask_c_flat[:, m]           # [N*T_seg]
            if not np.any(mask_m):
                freq_mqe[c, m] = 0.1
            else:
                vals = Y_c_flat[mask_m, m]       # values where observed (0 or 1)
                freq_mqe[c, m] = vals.mean()

    freq_mqe = np.clip(freq_mqe, 0.05, 0.95)   # avoid extremes
    # Beta_mqe[c, s, m]; start with same freq for all segment labels s
    Beta_mqe = np.zeros((C_mqe, K, M_mqe))
    for c in range(C_mqe):
        for s in range(K):
            Beta_mqe[c, s, :] = freq_mqe[c, :]

    # ---- precompute all segment-label combos (S vectors) ----
    svec_list = list(itertools.product(range(K), repeat=T_seg))  # length K^T_seg
    S_all = len(svec_list)  # = K ** T_seg

    for it in range(max_iter):
        # ---------- E-step ----------
        gamma_T = np.zeros((N, K))            # p(T_i=k | data)
        tau_S = np.zeros((N, T_seg, K))       # p(S_it=s | data)
        counts_TS = np.zeros((T_seg, K, K))   # for Phi_t update: [t, k, s]

        # precompute logs
        log_Theta_task = np.log(Theta_task + eps)  # [C_task, K, K]
        log_Theta_seg = np.log(Theta_seg + eps)    # [C_seg, K, K]
        log_Phi_t = np.log(Phi_t + eps)            # [T_seg, K, K]
        log_Beta = np.log(Beta_mqe + eps)          # [C_mqe, K, M_mqe]
        log_1mB  = np.log(1 - Beta_mqe + eps)      # [C_mqe, K, M_mqe]

        for i in range(N):
            # --- log p(R_task_i | T_i=k) ---
            log_L_task_i = np.zeros(K)
            for c in range(C_task):
                if not mask_task[i, c]:
                    continue
                obs_label = R_task[i, c]
                # add log Theta_task[c, k, obs_label] for all k
                log_L_task_i += log_Theta_task[c][:, obs_label]

            # --- log p(segment ratings + MQEs | S_it=s) per (t,s) ---
            # log_L_t_i[t,s]
            log_L_t_i = np.zeros((T_seg, K))
            for t_idx in range(T_seg):
                for s in range(K):
                    ll = 0.0
                    # segment ratings
                    for c in range(C_seg):
                        if not mask_seg[i, t_idx, c]:
                            continue
                        obs_seg = R_seg[i, t_idx, c]
                        ll += log_Theta_seg[c][s, obs_seg]
                    # MQEs
                    for c in range(C_mqe):
                        for m in range(M_mqe):
                            if not mask_mqe[i, t_idx, c, m]:
                                continue
                            y = Y_mqe[i, t_idx, c, m]
                            if y == 1:
                                ll += log_Beta[c, s, m]
                            elif y == 0:
                                ll += log_1mB[c, s, m]
                    log_L_t_i[t_idx, s] = ll

            # --- joint log p(T_i=k, S_i* = svec | data_i) ---
            log_joint = np.full((K, S_all), -np.inf, dtype=float)

            for k in range(K):
                # base for each k
                base_k = np.log(pi[k] + eps) + log_L_task_i[k]
                for s_idx, svec in enumerate(svec_list):
                    val = base_k
                    # add segments contributions
                    for t_idx, s in enumerate(svec):
                        val += log_Phi_t[t_idx, k, s] + log_L_t_i[t_idx, s]
                    log_joint[k, s_idx] = val

            # normalize to get q_i(k, svec)
            max_log = log_joint.max()
            joint = np.exp(log_joint - max_log)
            joint_sum = joint.sum()
            if joint_sum <= 0:
                # safeguard: fallback to uniform if something went wrong
                joint = np.ones_like(joint) / (K * S_all)
            else:
                joint /= joint_sum

            # posterior over T
            gamma_T[i, :] = joint.sum(axis=1)   # sum over svec

            # posterior over S_it and counts_TS
            for s_idx, svec in enumerate(svec_list):
                # total weight for this S configuration (marginalizing T)
                w_tot = joint[:, s_idx].sum()
                for t_idx, s in enumerate(svec):
                    tau_S[i, t_idx, s] += w_tot
                    # accumulate T-S counts for Phi_t
                    for k in range(K):
                        counts_TS[t_idx, k, s] += joint[k, s_idx]

        # normalize tau_S per (i,t)
        tau_sum = tau_S.sum(axis=2, keepdims=True)  # [N, T_seg, 1]
        tau_sum[tau_sum == 0] = 1.0
        tau_S /= tau_sum

        # ---------- M-step ----------

        # pi
        N_eff = gamma_T.sum(axis=0)  # [K]
        pi_new = (N_eff + alpha) / (N + K * alpha)

        # Theta_task update
        Theta_task_new = np.zeros_like(Theta_task)
        for c in range(C_task):
            num = np.zeros((K, K))
            denom = np.zeros(K)
            for i in range(N):
                if not mask_task[i, c]:
                    continue
                obs_label = R_task[i, c]
                gamma_i = gamma_T[i, :]  # [K]
                denom += gamma_i
                num[:, obs_label] += gamma_i
            Theta_task_new[c] = (num + alpha) / (denom[:, None] + K * alpha)

        # Theta_seg update
        Theta_seg_new = np.zeros_like(Theta_seg)
        for c in range(C_seg):
            num = np.zeros((K, K))
            denom = np.zeros(K)
            for i in range(N):
                for t_idx in range(T_seg):
                    if not mask_seg[i, t_idx, c]:
                        continue
                    obs_seg = R_seg[i, t_idx, c]
                    tau_it = tau_S[i, t_idx, :]  # [K]
                    denom += tau_it
                    num[:, obs_seg] += tau_it
            Theta_seg_new[c] = (num + alpha) / (denom[:, None] + K * alpha)

        # Phi_t update from counts_TS
        Phi_t_new = np.zeros_like(Phi_t)
        for t_idx in range(T_seg):
            counts_t = counts_TS[t_idx]  # [K (T label), K (S label)]
            for k in range(K):
                row = counts_t[k, :]  # [K]
                denom_k = row.sum()
                Phi_t_new[t_idx, k, :] = (row + alpha) / (denom_k + K * alpha) if denom_k > 0 else Phi_t[t_idx, k, :]

        # Beta_mqe update
        Beta_mqe_new = np.zeros_like(Beta_mqe)
        for c in range(C_mqe):
            for s in range(K):
                num_sm = np.zeros(M_mqe)
                denom_sm = np.zeros(M_mqe)
                for i in range(N):
                    for t_idx in range(T_seg):
                        tau_it_s = tau_S[i, t_idx, s]
                        if tau_it_s <= 0:
                            continue
                        for m in range(M_mqe):
                            if not mask_mqe[i, t_idx, c, m]:
                                continue
                            y = Y_mqe[i, t_idx, c, m]
                            denom_sm[m] += tau_it_s
                            if y == 1:
                                num_sm[m] += tau_it_s
                # Beta posterior mean: Beta(alpha_beta + num, alpha_beta + denom-num)
                Beta_mqe_new[c, s, :] = (num_sm + alpha_beta) / (denom_sm + 2 * alpha_beta)
                # clip away from 0/1
                Beta_mqe_new[c, s, :] = np.clip(Beta_mqe_new[c, s, :], 0.01, 0.99)

        # convergence checks
        delta_pi = np.max(np.abs(pi_new - pi))
        delta_theta_task = np.max(np.abs(Theta_task_new - Theta_task))
        delta_theta_seg = np.max(np.abs(Theta_seg_new - Theta_seg))
        delta_phi = np.max(np.abs(Phi_t_new - Phi_t))
        delta_beta = np.max(np.abs(Beta_mqe_new - Beta_mqe))

        if verbose:
            print(f"[full T→S→MQE] Iter {it}: "
                  f"Δpi={delta_pi:.3e}, ΔTheta_task={delta_theta_task:.3e}, "
                  f"ΔTheta_seg={delta_theta_seg:.3e}, ΔPhi={delta_phi:.3e}, "
                  f"ΔBeta={delta_beta:.3e}")

        pi, Theta_task, Theta_seg, Phi_t, Beta_mqe = (
            pi_new, Theta_task_new, Theta_seg_new, Phi_t_new, Beta_mqe_new
        )

        if max(delta_pi, delta_theta_task, delta_theta_seg, delta_phi, delta_beta) < tol:
            if verbose:
                print("[full T→S→MQE] Converged.")
            break

    return gamma_T, tau_S, pi, Theta_task, Theta_seg, Phi_t, Beta_mqe

if __name__ == "__main__":
    """
    Example with fake shapes matching your counts:

    N_instances ~ 1800
    T_seg = 4
    C = 2 clinicians
    M_mqe (e.g., 10 movement quality elements)
    n_task_types = 15 ARAT items

    Replace the fake random arrays below with your actual annotations.
    """

    np.random.seed(0)

    n_draws = 100
    n_tunes = 100
    n_chains = 2
    T_seg = 4
    C = 2
    M_mqe = 10           # adjust to your actual number of MQE types
    n_task_types = 15
    K = 4                # ARAT scores 0..3
    csv_dir = r"D:\nature_everything"
    records = load_video_segments_info(csv_dir)
    task_ratings_dict, segment_ratings_dict, composite_rating_dict = load_rating_info(csv_dir)
    N_instances = len(task_ratings_dict)
    # Task type for each instance (0..14)
    task_type_id = np.random.randint(0, n_task_types, size=N_instances)
    # 1) Decide clinician order (columns)
    clinician_keys = ["t1", "t2"]   # col 0 = t1, col 1 = t2
    C = len(clinician_keys)

    # 2) Fix a deterministic ordering of instances
    #    (this index i will be row i in R_task)
    instance_ids = sorted(task_ratings_dict.keys())  # e.g., [7, 43, 83, 90, ...]
    N_instances = len(instance_ids)

    # 3) Create R_task with default -1 (missing)
    R_task = np.full((N_instances, C), -1, dtype=int)

    # 4) Optional: keep mappings between row index <-> instance id
    idx_to_instance_id = instance_ids
    instance_id_to_idx = {inst_id: i for i, inst_id in enumerate(instance_ids)}

    # 5) Fill R_task from the dict
    for i, inst_id in enumerate(instance_ids):
        rating_dict = task_ratings_dict[inst_id]  # e.g. {'t1': 0, 't2': 1}
        for c, ck in enumerate(clinician_keys):
            if ck in rating_dict and rating_dict[ck] is not None:
                # assumes rating is already an int in {0,1,2,3}
                R_task[i, c] = int(rating_dict[ck])
            # else: leave as -1 (no rating from that clinician)

    print("R_task shape:", R_task.shape)
    print("First few rows of R_task:\n", R_task[:5])
    print("First few instance ids:", idx_to_instance_id[:5])



    valid_segments = {1, 2, 3, 4}
    T_seg = 4  # we want exactly 4 segments per instance

    # Map segment numbers {1,2,3,4} -> indices {0,1,2,3}
    seg_index = {1: 0, 2: 1, 3: 2, 4: 3}

    N_instances = len(idx_to_instance_id)

    # Initialize R_seg with -1 (missing)
    R_seg = np.full((N_instances, T_seg, C), -1, dtype=int)

    # -------------------------------------------------
    # Fill R_seg from segment_rating_dict
    # -------------------------------------------------
    for i, inst_id in enumerate(idx_to_instance_id):
        rating_entry = segment_ratings_dict.get(inst_id, None)

        # If we have no segment info for this instance, leave as all -1
        if rating_entry is None:
            continue

        # Collect all segment ids used by any clinician for this instance
        all_seg_ids = set()
        for ck in clinician_keys:
            seg_dict = rating_entry.get(ck, {})
            for s in seg_dict.keys():
                # segment keys are floats like 1.0, 2.0 -> cast to int
                all_seg_ids.add(int(s))

        # If any segment id is not in {1,2,3,4}, ignore this instance's segments
        if not all(s in valid_segments for s in all_seg_ids):
            # Leave R_seg[i, :, :] as -1
            continue

        # Otherwise, fill in ratings for segments 1..4 (missing ones stay -1)
        for c, ck in enumerate(clinician_keys):
            seg_dict = rating_entry.get(ck, {})
            for s, score in seg_dict.items():
                s_int = int(s)
                if s_int in seg_index:
                    t_idx = seg_index[s_int]
                    R_seg[i, t_idx, c] = int(score)

    print("R_seg shape:", R_seg.shape)
    print("Example row (instance 0):")
    print(R_seg[0])
    
    post_task, pi_task, Theta_task = ds_multiclass(R_task, K=K, max_iter=500, verbose=True)

    # Uncertainty per task
    H_task = entropy(post_task)  # [N]
    print("Mean task entropy (task-only model):", H_task.mean())

    # MAP task predictions if you want
    T_hat = post_task.argmax(axis=1)


    gamma_T, q_S, pi_ts, Theta_task_ts, Theta_seg_ts, Phi_t = em_task_segment(
        R_task,
        R_seg,
        K=K,
        max_iter=500,
        verbose=True,
    )

    # Task entropy with segment info
    H_task_ts = entropy(gamma_T)  # [N]

    print("Mean task entropy (task-only DS):", entropy(ds_multiclass(R_task, K=4)[0]).mean())
    print("Mean task entropy (task+segment model):", H_task_ts.mean())
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))

    bins = 30
    plt.hist(H_task, bins=bins, alpha=0.5, label="Task-only")
    plt.hist(H_task_ts,  bins=bins, alpha=0.5, label="Task + segments")

    plt.xlabel("Task entropy H(T)")
    plt.ylabel("Number of tasks")
    plt.title("Entropy distribution: task-only vs task+segment")
    plt.legend()
    plt.tight_layout()
    plt.show()
    delta_H = H_task - H_task_ts  # >0 means uncertainty reduced

    plt.figure(figsize=(8, 5))
    plt.hist(delta_H, bins=30)
    plt.xlabel("Entropy reduction ΔH = H_task_only - H_task_seg")
    plt.ylabel("Number of tasks")
    plt.title("Per-task entropy reduction from adding segments")
    plt.tight_layout()
    plt.show()

    print("Fraction of tasks with ΔH > 0:", (delta_H > 0).mean())

    plt.figure(figsize=(6, 6))
    plt.scatter(H_task, H_task_ts, s=10)

    # Reference line y = x (no change)
    min_val = min(H_task.min(), H_task_ts.min())
    max_val = max(H_task.max(), H_task_ts.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    plt.xlabel("Entropy (task-only)")
    plt.ylabel("Entropy (task+segments)")
    plt.title("Per-task entropy: task-only vs task+segments")
    plt.tight_layout()
    plt.show()

    hard_mask = H_task > 0.8  # choose a threshold you like
    print("Number of 'hard' tasks:", hard_mask.sum())

    plt.figure(figsize=(8, 5))
    plt.hist(H_task[hard_mask], bins=20, alpha=0.5, label="Task-only (hard subset)")
    plt.hist(H_task_ts[hard_mask],  bins=20, alpha=0.5, label="Task+segments (hard subset)")

    plt.xlabel("Task entropy H(T) (hard subset)")
    plt.ylabel("Number of tasks")
    plt.title("Effect of segments on high-uncertainty tasks")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print('done')


    # 1) Build MQE arrays
    M_mqe = None  # infer automatically; or set =14
    X_mqe_task, X_mqe_seg = build_mqe_arrays(
        composite_rating_dict,
        idx_to_instance_id,
        T_seg=4,
        clinician_keys=("t1", "t2"),
        M_mqe=M_mqe,
    )    
    # 3) Task + MQE model
    gamma_T_mqe, pi_mqe, Theta_task_mqe, Beta_task = em_task_mqe(
        R_task,
        X_mqe_task,
        K=K,
        max_iter=50,
        verbose=True,
    )
    H_task_mqe = entropy(gamma_T_mqe)
    # 4) Task + segment + MQE model
    gamma_T_seg_mqe, pi_seg_mqe, Theta_task_seg_mqe, Theta_seg_seg_mqe, Beta_seg = em_task_seg_mqe(
        R_task,
        R_seg,
        X_mqe_seg,
        K=K,
        max_iter=50,
        verbose=True,
    )
    H_task_seg_mqe = entropy(gamma_T_seg_mqe)

    print("Mean entropy (task-only):        ", H_task.mean())
    print("Mean entropy (task+MQE):         ", H_task_mqe.mean())
    print("Mean entropy (task+seg+MQE):     ", H_task_seg_mqe.mean())    

    # Overlaid histograms
    plt.figure(figsize=(8, 5))
    bins = 30
    plt.hist(H_task,   bins=bins, alpha=0.4, label="Task-only")
    plt.hist(H_task_mqe,    bins=bins, alpha=0.4, label="Task + MQE")
    plt.hist(H_task_seg_mqe,bins=bins, alpha=0.4, label="Task + seg + MQE")
    plt.xlabel("Task entropy H(T)")
    plt.ylabel("Number of tasks")
    plt.title("Entropy distributions: task-only vs task+MQE vs task+seg+MQE")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Entropy reductions: task-only -> task+MQE, task+MQE -> task+seg+MQE
    delta_H_task_to_mqe = H_task - H_task_mqe
    delta_H_mqe_to_seg  = H_task_mqe - H_task_seg_mqe

    plt.figure(figsize=(8, 5))
    plt.hist(delta_H_task_to_mqe, bins=30, alpha=0.6, label="ΔH (task-only → task+MQE)")
    plt.hist(delta_H_mqe_to_seg,  bins=30, alpha=0.6, label="ΔH (task+MQE → task+seg+MQE)")
    plt.axvline(0.0, color="k", linestyle="--")
    plt.xlabel("Entropy change ΔH")
    plt.ylabel("Number of tasks")
    plt.title("Entropy reduction per task")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("Frac with ΔH(task→MQE) > 0:", (delta_H_task_to_mqe > 0).mean())
    print("Frac with ΔH(MQE→seg+MQE) > 0:", (delta_H_mqe_to_seg > 0).mean())



    # 1) Build MQE arrays for the full model
    Y_mqe = build_mqe_rater_arrays(
        composite_rating_dict,
        idx_to_instance_id,
        T_seg=4,
        clinician_keys=("t1", "t2"),
        M_mqe=None,    # infer 14 from dict
    )

    # 2) Existing models (assuming you already ran them):
    post_task_only, _, _ = ds_multiclass(R_task, K=K)
    H_task_only = entropy(post_task_only)

    gamma_T_mqe, _, _, _ = em_task_mqe(R_task, X_mqe_task, K=K)
    H_task_mqe = entropy(gamma_T_mqe)

    gamma_T_seg_mqe, _, _, _, _ = em_task_seg_mqe(R_task, R_seg, X_mqe_seg, K=K)
    H_task_seg_mqe = entropy(gamma_T_seg_mqe)

    # 3) Full hierarchical T->S->MQE model
    gamma_T_full, tau_S_full, pi_full, Theta_task_full, Theta_seg_full, Phi_t_full, Beta_mqe_full = \
        em_full_TS_MQE(R_task, R_seg, Y_mqe, K=K, max_iter=30, verbose=True)

    H_task_full = entropy(gamma_T_full)        # [N]
    H_seg_full = entropy(tau_S_full)          # [N, T_seg]
    H_seg_mean = H_seg_full.mean(axis=1)      # mean segment entropy per task

    print("Mean entropy (task-only):          ", H_task_only.mean())
    print("Mean entropy (task+MQE flat):      ", H_task_mqe.mean())
    print("Mean entropy (task+seg+MQE flat):  ", H_task_seg_mqe.mean())
    print("Mean entropy (full T→S→MQE):       ", H_task_full.mean())



    plt.figure(figsize=(8, 5))
    bins = 30

    plt.hist(H_task_only,    bins=bins, alpha=0.4, label="Task-only")
    plt.hist(H_task_mqe,     bins=bins, alpha=0.4, label="Task + MQE")
    plt.hist(H_task_seg_mqe, bins=bins, alpha=0.4, label="Task + seg + MQE (flat)")
    plt.hist(H_task_full,    bins=bins, alpha=0.4, label="Full T→S→MQE")

    plt.xlabel("Task entropy H(T)")
    plt.ylabel("Number of tasks")
    plt.title("Task entropy distributions across label models")
    plt.legend()
    plt.tight_layout()
    plt.show()

    delta_full_vs_flat_seg = H_task_seg_mqe - H_task_full

    plt.figure(figsize=(8, 5))
    plt.hist(delta_full_vs_flat_seg, bins=30, alpha=0.7)
    plt.axvline(0.0, color="k", linestyle="--")
    plt.xlabel("ΔH = H(task+seg+MQE flat) - H(full T→S→MQE)")
    plt.ylabel("Number of tasks")
    plt.title("Entropy reduction when using full hierarchical T→S→MQE")
    plt.tight_layout()
    plt.show()

    print("Fraction of tasks with entropy reduced by full hierarchy:",
        (delta_full_vs_flat_seg > 0).mean())
    # Mean segment-level entropy per task
    plt.figure(figsize=(8, 5))
    plt.scatter(H_task_full, H_seg_mean, s=10)
    plt.xlabel("Task entropy (full T→S→MQE)")
    plt.ylabel("Mean segment entropy per task")
    plt.title("Relation between task- and segment-level uncertainty")
    plt.tight_layout()
    plt.show()
    print('done')
