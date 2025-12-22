import numpy as np
import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path
EPS = 1e-8

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


def load_rating_info(csv_dir):
    """
    Reads task_final.csv and segment_final.csv and returns:
        task_ratings       {mapping_id: {'t1': r, 't2': r2}}
        segment_ratings    {mapping_id: {'t1': {seg: r, ...}, 't2': {...}}}
        composite_ratings  {mapping_id: {'t1': {seg: '0,1,...'}, 't2': {...}}}
    The lower TherapistId per mapping is always “t1”, higher (if present) is “t2”.
    """
    # task_df    = pd.read_csv(os.path.join(csv_dir, "task_final.csv"))
    # seg_df     = pd.read_csv(os.path.join(csv_dir, "segment_final.csv"))
    task_df    = pd.read_csv(os.path.join(csv_dir, "task_final_updated.csv"))
    seg_df     = pd.read_csv(os.path.join(csv_dir, "segment_final_updated.csv"))
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

    Segment-aware: only MQEs relevant to each segment (per SEGMENT_COMPOSITES)
    contribute to the likelihood and Beta updates.

    Inputs:
      R_task:    [N, C_task]        int in 0..K-1 or -1
      R_seg:     [N, T_seg, C_seg]  int in 0..K-1 or -1
      X_mqe_seg: [N, T_seg, M]      binary (0/1)

    Returns:
      gamma_T:    [N, K]       posterior p(T_i=k | all)
      pi:         [K]
      Theta_task: [C_task, K, K]
      Theta_seg:  [C_seg, K, K]
      Beta_seg:   [K, M]
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

    # --- build segment→MQE mask ---
    seg_mqe_mask = build_segment_mqe_mask(T_seg, M)  # [T_seg, M] bool
    # indices of relevant MQEs per segment
    mqe_idx_per_seg = [np.where(seg_mqe_mask[t])[0] for t in range(T_seg)]
    # how many segments each MQE participates in
    seg_counts_per_m = seg_mqe_mask.sum(axis=0)  # [M]

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

    # ---- init Beta_seg from global MQE frequencies (using all segments) ----
    freq_mqe_seg = X_mqe_seg.reshape(N * T_seg, M).mean(axis=0)
    freq_mqe_seg = np.clip(freq_mqe_seg, 0.05, 0.95)
    Beta_seg = np.tile(freq_mqe_seg[None, :], (K, 1))  # [K, M]

    for it in range(max_iter):
        # ---------- E-step ----------
        log_pi = np.log(pi + eps)
        log_Theta_task = np.log(Theta_task + eps)  # [C_task, K, K]
        log_Theta_seg = np.log(Theta_seg + eps)    # [C_seg, K, K]
        log_Beta = np.log(Beta_seg + eps)          # [K, M]
        log_1mB  = np.log(1 - Beta_seg + eps)      # [K, M]

        # log p(R_task | T_i=k)
        log_p_Rtask = np.zeros((N, K))
        for c in range(C_task):
            obs_idx = mask_task[:, c]
            if not np.any(obs_idx):
                continue
            labels_c = R_task[obs_idx, c]
            # log_Theta_task[c]: [K, K], select columns by labels_c
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
            labels = R_seg[i_idx, t_idx, c]  # [n_pairs]
            log_theta_c = log_Theta_seg[c]   # [K, K]
            contrib = log_theta_c[:, labels].T  # [n_pairs, K]
            for row, ii in enumerate(i_idx):
                log_p_Rseg[ii, :] += contrib[row]

        # log p(X_mqe_seg | T_i=k)  (segment-aware MQEs)
        log_p_MQE_seg = np.zeros((N, K))
        for k in range(K):
            for i in range(N):
                lp = 0.0
                for t in range(T_seg):
                    idxs = mqe_idx_per_seg[t]
                    if idxs.size == 0:
                        continue
                    xm = X_mqe_seg[i, t, idxs]  # [len(idxs)]
                    lp += (xm * log_Beta[k, idxs] +
                           (1 - xm) * log_1mB[k, idxs]).sum()
                log_p_MQE_seg[i, k] = lp

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

        # Beta_seg (segment-aware)
        Beta_seg_new = np.zeros_like(Beta_seg)
        for k in range(K):
            gamma_k = gamma_T[:, k]  # [N]
            # For each MQE m, only segments where seg_mqe_mask[t,m] is True contribute.
            num_k = np.zeros(M)
            denom_k = np.zeros(M)
            sum_gamma_k = gamma_k.sum()

            for t in range(T_seg):
                idxs = mqe_idx_per_seg[t]
                if idxs.size == 0:
                    continue
                # X slice for these MQEs on this segment
                X_slice = X_mqe_seg[:, t, idxs]  # [N, len(idxs)]
                # numerator: sum_i gamma_k[i] * X_mqe_seg[i,t,m]
                num_k[idxs] += (gamma_k[:, None] * X_slice).sum(axis=0)
                # denominator: sum_i gamma_k[i] for each relevant segment
                denom_k[idxs] += sum_gamma_k

            # finalize Beta for this class k
            for m in range(M):
                if denom_k[m] <= 0 or seg_counts_per_m[m] == 0:
                    # no information for this MQE / class; keep old value
                    Beta_seg_new[k, m] = Beta_seg[k, m]
                else:
                    Beta_seg_new[k, m] = (num_k[m] + alpha_beta) / (denom_k[m] + 2 * alpha_beta)

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


def em_full_TS_MQE_meanfield(
    R_task,
    R_seg,
    Y_mqe,
    K=4,
    max_iter=30,
    tol=1e-5,
    alpha=1e-2,
    alpha_beta=1e-2,
    verbose=False,
):
    """
    Unconstrained mean-field EM for hierarchical T→S→MQE model with segment-aware MQEs.

      Latent:
        T_i ∈ {0,1,2,3}          (task score)
        S_{i,t} ∈ {0,1,2,3}      (segment score for t=1..T_seg)

      Observed:
        R_task[i,c]      ~ Cat(Theta_task[c, T_i, :])
        R_seg[i,t,c]     ~ Cat(Theta_seg[c, S_{i,t}, :])
        Y_mqe[i,t,c,m]   ~ Bernoulli(beta_mqe[c, S_{i,t}, m])
                          but only for segment-relevant MQEs (via SEGMENT_COMPOSITES)

      Transitions:
        S_{i,t} | T_i=k  ~ Cat(Phi[t, k, :])

      Mean-field approximation:
        q(T_i, S_{i,1:T_seg}) ≈ q(T_i) ∏_t q(S_{i,t})

    Inputs
    ------
      R_task : (N, C_task) in {0,1,2,3,-1}
      R_seg  : (N, T_seg, C_seg) in {0,1,2,3,-1}
      Y_mqe  : (N, T_seg, C_mqe, M) in {0,1,-1}

    Returns
    -------
      gamma_T     : (N, K)       posterior q(T_i)
      tau_S       : (N, T_seg, K) posterior q(S_{i,t})
      pi          : (K,)
      Phi         : (T_seg, K, K)
      Theta_task  : (C_task, K, K)
      Theta_seg   : (C_seg, K, K)
      beta_mqe    : (C_mqe, K, M)
    """
    EPS = 1e-12

    R_task = np.asarray(R_task, dtype=int)
    R_seg  = np.asarray(R_seg,  dtype=int)
    Y_mqe  = np.asarray(Y_mqe,  dtype=int)

    N, C_task = R_task.shape
    N2, T_seg, C_seg = R_seg.shape
    N3, T_seg2, C_mqe, M = Y_mqe.shape

    assert N2 == N
    assert N3 == N
    assert T_seg2 == T_seg

    mask_task = R_task >= 0
    mask_seg  = R_seg  >= 0

    # --- build segment→MQE mask + indices ---
    seg_mqe_mask = build_segment_mqe_mask(T_seg, M)    # (T_seg, M) bool
    mqe_idx_per_seg = [np.where(seg_mqe_mask[t])[0] for t in range(T_seg)]

    # ---------- Init ----------

    # pi from task labels
    labels_flat = R_task[mask_task]
    if labels_flat.size > 0:
        counts = np.bincount(labels_flat, minlength=K).astype(float)
        if counts.sum() > 0:
            pi = counts / counts.sum()
        else:
            pi = np.ones(K) / K
    else:
        pi = np.ones(K) / K

    # Theta_task, Theta_seg near-identity
    Theta_task = np.zeros((C_task, K, K))
    Theta_seg  = np.zeros((C_seg,  K, K))
    for c in range(C_task):
        Theta_task[c] = np.eye(K) * 0.9 + (1 - np.eye(K)) * 0.1 / (K - 1)
    for c in range(C_seg):
        Theta_seg[c]  = np.eye(K) * 0.9 + (1 - np.eye(K)) * 0.1 / (K - 1)

    # Phi (T→S): diag-heavy per segment
    Phi = np.full((T_seg, K, K), 1.0 / K)
    for t in range(T_seg):
        for k in range(K):
            Phi[t, k, k] = 0.6
            if k > 0:
                Phi[t, k, k - 1] = 0.2
            if k < K - 1:
                Phi[t, k, k + 1] = 0.2
        Phi[t] /= Phi[t].sum(axis=1, keepdims=True)
    # log_Phi updated each iteration after M-step
    log_Phi = np.log(Phi + EPS)

    # beta_mqe: (C_mqe, K, M) init from global frequencies over all segments
    base_counts = (Y_mqe == 1).sum(axis=(0, 1))       # (C_mqe, M)
    base_obs    = (Y_mqe != -1).sum(axis=(0, 1)) + EPS
    base_rate   = np.clip(base_counts / base_obs, 0.05, 0.95)
    beta_mqe    = np.repeat(base_rate[:, None, :], K, axis=1)

    # mean-field posteriors
    gamma_T = np.full((N, K), 1.0 / K)        # q(T_i)
    tau_S   = np.full((N, T_seg, K), 1.0 / K) # q(S_{i,t})

    for it in range(max_iter):
        # ============================
        # E-STEP
        # ============================

        # Precompute logs for current parameters
        log_pi         = np.log(pi + EPS)
        log_Theta_task = np.log(Theta_task + EPS)  # (C_task, K, K)
        log_Theta_seg  = np.log(Theta_seg  + EPS)  # (C_seg,  K, K)
        log_beta       = np.log(beta_mqe + EPS)    # (C_mqe, K, M)
        log_1m_beta    = np.log(1.0 - beta_mqe + EPS)

        # --- Task-level likelihood: ll_task[i, k] ---
        ll_task = np.zeros((N, K), dtype=float)
        for c in range(C_task):
            obs_idx = mask_task[:, c]
            if not np.any(obs_idx):
                continue
            labels_c = R_task[obs_idx, c]          # observed labels for clinician c
            # log_Theta_task[c]: (K, K)  -> columns by labels_c
            ll_task[obs_idx, :] += log_Theta_task[c][:, labels_c].T

        # --- Segment + MQE likelihoods: ll_seg[i,t,s], ll_mqe[i,t,s] ---
        ll_seg = np.zeros((N, T_seg, K), dtype=float)
        ll_mqe = np.zeros((N, T_seg, K), dtype=float)

        # Segment ratings
        for c in range(C_seg):
            obs_idx_it = mask_seg[:, :, c]          # (N, T_seg)
            idx_pairs = np.argwhere(obs_idx_it)
            if idx_pairs.size == 0:
                continue
            i_idx = idx_pairs[:, 0]
            t_idx = idx_pairs[:, 1]
            labels = R_seg[i_idx, t_idx, c]         # (n_pairs,)
            log_theta_c = log_Theta_seg[c]          # (K, K)
            contrib = log_theta_c[:, labels].T      # (n_pairs, K)
            for row, ii in enumerate(i_idx):
                tt = t_idx[row]
                ll_seg[ii, tt, :] += contrib[row]

        # MQE likelihoods, segment-aware
        for i in range(N):
            for t in range(T_seg):
                idxs = mqe_idx_per_seg[t]
                if idxs.size == 0:
                    continue
                for s in range(K):
                    v_mqe = 0.0
                    for c in range(C_mqe):
                        for m in idxs:
                            y = Y_mqe[i, t, c, m]
                            if y == -1:
                                continue
                            p_log   = log_beta[c, s, m]
                            q_log   = log_1m_beta[c, s, m]
                            if y == 1:
                                v_mqe += p_log
                            else:
                                v_mqe += q_log
                    ll_mqe[i, t, s] = v_mqe

        # --- Update tau_S (q(S_{i,t})) ---
        for i in range(N):
            for t in range(T_seg):
                # message from T to S: E_{q_T}[log p(S_t | T)]
                msg_T_to_S = np.zeros(K, dtype=float)
                for s in range(K):
                    msg = 0.0
                    for k in range(K):
                        msg += gamma_T[i, k] * log_Phi[t, k, s]
                    msg_T_to_S[s] = msg

                log_tau = msg_T_to_S + ll_seg[i, t] + ll_mqe[i, t]
                m = np.max(log_tau)
                x = log_tau - m
                tau_S[i, t] = np.exp(x) / (np.sum(np.exp(x)) + EPS)

        # --- Update gamma_T (q(T_i)) ---
        log_gamma = np.zeros((N, K), dtype=float)
        for i in range(N):
            for k in range(K):
                val = log_pi[k] + ll_task[i, k]
                # sum over segments: E_{q_S}[log p(S_t | T=k)]
                for t in range(T_seg):
                    for s in range(K):
                        val += tau_S[i, t, s] * log_Phi[t, k, s]
                log_gamma[i, k] = val

        for i in range(N):
            m = np.max(log_gamma[i])
            x = log_gamma[i] - m
            gamma_T[i] = np.exp(x) / (np.sum(np.exp(x)) + EPS)

        # ============================
        # M-STEP
        # ============================

        Nk = gamma_T.sum(axis=0) + EPS    # (K,)

        # pi
        pi_new = (Nk + alpha) / (N + K * alpha)

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

        # Theta_seg (uses tau_S)
        Theta_seg_new = np.zeros_like(Theta_seg)
        for c in range(C_seg):
            num = np.zeros((K, K))
            denom = np.zeros(K)
            for i in range(N):
                for t in range(T_seg):
                    if not mask_seg[i, t, c]:
                        continue
                    l = R_seg[i, t, c]
                    denom += tau_S[i, t, :]
                    num[:, l] += tau_S[i, t, :]
            Theta_seg_new[c] = (num + alpha) / (denom[:, None] + K * alpha)

        # Phi (T→S)
        Phi_new = np.zeros_like(Phi) + EPS
        for t in range(T_seg):
            for k in range(K):
                for s in range(K):
                    Phi_new[t, k, s] += np.sum(gamma_T[:, k] * tau_S[:, t, s])
        # normalize over s
        Phi_new /= Phi_new.sum(axis=2, keepdims=True)
        log_Phi_new = np.log(Phi_new + EPS)

        # beta_mqe: (C_mqe, K, M), segment-aware
        beta_mqe_new = np.zeros_like(beta_mqe)
        denom = np.zeros_like(beta_mqe)

        for i in range(N):
            for t in range(T_seg):
                idxs = mqe_idx_per_seg[t]
                if idxs.size == 0:
                    continue
                for c in range(C_mqe):
                    for m in idxs:
                        y = Y_mqe[i, t, c, m]
                        if y == -1:
                            continue
                        for s in range(K):
                            w = tau_S[i, t, s]
                            if y == 1:
                                beta_mqe_new[c, s, m] += w
                            denom[c, s, m] += w

        beta_mqe_new = (beta_mqe_new + alpha_beta) / (denom + 2 * alpha_beta)
        beta_mqe_new = np.clip(beta_mqe_new, 0.01, 0.99)

        # convergence diagnostics
        delta_pi    = np.max(np.abs(pi_new - pi))
        delta_ttask = np.max(np.abs(Theta_task_new - Theta_task))
        delta_tseg  = np.max(np.abs(Theta_seg_new  - Theta_seg))
        delta_phi   = np.max(np.abs(Phi_new       - Phi))
        delta_beta  = np.max(np.abs(beta_mqe_new  - beta_mqe))

        if verbose:
            print(f"[full T→S→MQE mean-field] Iter {it}: "
                  f"Δpi={delta_pi:.3e}, "
                  f"ΔTheta_task={delta_ttask:.3e}, "
                  f"ΔTheta_seg={delta_tseg:.3e}, "
                  f"ΔPhi={delta_phi:.3e}, "
                  f"ΔBeta={delta_beta:.3e}")

        pi, Theta_task, Theta_seg, Phi, beta_mqe = \
            pi_new, Theta_task_new, Theta_seg_new, Phi_new, beta_mqe_new
        log_Phi = log_Phi_new

        if max(delta_pi, delta_ttask, delta_tseg, delta_phi, delta_beta) < tol:
            if verbose:
                print("[full T→S→MQE mean-field] Converged.")
            break

    return gamma_T, tau_S, pi, Phi, Theta_task, Theta_seg, beta_mqe



# Global MQE ordering (index 0..13)
MQE_NAMES = [
    'SEAFR', 'TS', 'ROME', 'FPS',
    'WPAT', 'HA', 'DP', 'DPO',
    'SAT', 'DMR', 'THS', 'PP',
    'FPO', 'SA'
]
# Segment → relevant MQEs (by name)
SEGMENT_COMPOSITES = {
    0: ['SEAFR', 'TS', 'ROME', 'FPS'],      # IP segment
    1: ['WPAT', 'HA', 'DP', 'SA'],          # T segment
    2: ['SAT', 'FPS', 'TS', 'DPO'],         # MTR segment
    3: ['FPO', 'DMR', 'TS', 'FPS']          # PR segment
}


def build_segment_mqe_mask(T_seg, M):
    """
    Build a boolean mask of shape (T_seg, M) where mask[t, m] is True
    iff MQE m is clinically relevant for segment t.

    Assumes:
      - T_seg >= 4 (we use first 4 segments for this mapping)
      - M == 14 and MQE_NAMES gives the dimension order
    """
    name2idx = {name: i for i, name in enumerate(MQE_NAMES)}

    mask = np.zeros((T_seg, M), dtype=bool)
    for t, names in SEGMENT_COMPOSITES.items():
        if t >= T_seg:
            continue
        for nm in names:
            idx = name2idx[nm]
            if idx < M:
                mask[t, idx] = True
    return mask


def compute_entropy(probs, axis=-1, eps=1e-8):
    p = np.clip(probs, eps, 1.0)
    return -np.sum(p * np.log(p), axis=axis)
def compute_segment_entropy(tau_S):
    """
    tau_S: [N, T_seg, K]  posterior q(S_{i,t})

    Returns:
      seg_entropy      : [N, T_seg]  entropy per segment
      mean_seg_entropy : [N]         mean entropy across segments for each task
    """
    seg_entropy = compute_entropy(tau_S, axis=2)  # [N, T_seg]
    mean_seg_entropy = seg_entropy.mean(axis=1)   # [N]
    return seg_entropy, mean_seg_entropy


def compute_mqe_posterior_probs(tau_S, beta_mqe):
    """
    tau_S    : [N, T_seg, K]
    beta_mqe : [C_mqe, K, M]

    Returns
    -------
      q_mqe_seg    : [N, T_seg, M]
      q_mqe_any    : [N, M]
      seg_mqe_mask : [T_seg, M]
    """
    N, T_seg, K  = tau_S.shape
    C_mqe, K2, M = beta_mqe.shape
    assert K2 == K

    seg_mqe_mask = build_segment_mqe_mask(T_seg, M)   # <-- key line
    mqe_idx_per_seg = [np.where(seg_mqe_mask[t])[0] for t in range(T_seg)]

    beta_mean = beta_mqe.mean(axis=0)  # [K, M]
    q_mqe_seg = np.zeros((N, T_seg, M), dtype=float)

    for t in range(T_seg):
        idxs = mqe_idx_per_seg[t]
        if idxs.size == 0:
            continue
        for i in range(N):
            tau_it = tau_S[i, t, :]               # [K]
            q_mqe_seg[i, t, idxs] = tau_it @ beta_mean[:, idxs]

    # any-segment probability per MQE
    q_mqe_any = 1.0 - np.prod(1.0 - q_mqe_seg, axis=1)  # [N, M]

    return q_mqe_seg, q_mqe_any, seg_mqe_mask


def explain_trial(i, gamma_T, tau_S, q_mqe_seg, q_mqe_any,
                  mqe_names, top_k_mqe=5):
    """
    Human-readable explanation of *why* a trial is impaired,
    in terms of segments and MQEs.

    gamma_T  : [N, K]
    tau_S    : [N, T_seg, K]
    q_mqe_seg: [N, T_seg, M]
    q_mqe_any: [N, M]
    """
    K = gamma_T.shape[1]
    N, T_seg, _ = tau_S.shape
    _, M = q_mqe_any.shape

    assert 0 <= i < N

    probs_T = gamma_T[i]
    pred_T  = probs_T.argmax()
    ent_T   = compute_entropy(probs_T)

    print(f"=== Trial {i} ===")
    print(f"Task posterior: {probs_T} (entropy={ent_T:.4f})")
    print(f"Predicted task score (MAP): {pred_T}")

    # segments
    seg_entropy, _ = compute_segment_entropy(tau_S)
    print("\nSegments:")
    for t in range(T_seg):
        probs_S = tau_S[i, t, :]
        pred_S  = probs_S.argmax()
        ent_S   = seg_entropy[i, t]
        print(f"  Segment {t}: p(S)={probs_S}, MAP={pred_S}, H={ent_S:.4f}")

    # MQEs (any segment)
    print("\nTop MQEs (any segment):")
    probs_m = q_mqe_any[i]        # [M]
    order = np.argsort(-probs_m)  # descending
    for rank in range(min(top_k_mqe, M)):
        m = order[rank]
        print(f"  {mqe_names[m]}: P(any segment impaired)={probs_m[m]:.3f}")

    # Segment-specific MQEs
    print("\nSegment-specific MQEs (top per segment):")
    for t in range(T_seg):
        probs_m_seg = q_mqe_seg[i, t, :]
        order = np.argsort(-probs_m_seg)
        print(f"  Segment {t}:")
        for rank in range(min(top_k_mqe, M)):
            m = order[rank]
            if probs_m_seg[m] < 1e-3:
                break
            print(f"    {mqe_names[m]}: P={probs_m_seg[m]:.3f}")


from sklearn.cluster import KMeans

def cluster_mqe_patterns_for_label(
    gamma_T,
    q_mqe_any,
    target_label=2,
    min_conf=0.8,
    n_clusters=3,
    random_state=0
):
    """
    Cluster MQE patterns among trials whose MAP task == target_label
    AND p(target_label) >= min_conf.

    Returns
    -------
      idx_selected : indices of trials used
      cluster_ids  : same length as idx_selected
      centers      : [n_clusters, M] cluster centroids in MQE-prob space
    """
    N, K = gamma_T.shape
    N2, M = q_mqe_any.shape
    assert N == N2

    # select high-confidence trials of target_label
    map_labels = gamma_T.argmax(axis=1)
    conf       = gamma_T[np.arange(N), map_labels]
    mask = (map_labels == target_label) & (conf >= min_conf)
    idx_selected = np.where(mask)[0]

    if idx_selected.size == 0:
        raise ValueError("No high-confidence trials for the given label / min_conf.")

    X = q_mqe_any[idx_selected]  # [n_sel, M]

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    cluster_ids = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_   # [n_clusters, M]

    return idx_selected, cluster_ids, centers

def summarize_mechanistic_uncertainty_full(
    gamma_T, tau_S, q_mqe_any, seg_mqe_mask, verbose=True
):
    """
    Compute decomposition / mechanistic uncertainty, ignoring MQEs that
    are not used in any segment (e.g., THS, PP in this setup).

    Inputs
    ------
      gamma_T      : [N, K]        posterior over task labels
      tau_S        : [N, T_seg, K] posterior over segment labels
      q_mqe_any    : [N, M]        P(MQE_m=1 in any segment | data)
      seg_mqe_mask : [T_seg, M]    True where MQE_m is used in segment t

    Returns
    -------
      summary : dict of means/stds
      details : dict of per-trial arrays
    """
    eps = 1e-12

    # ---------- 1. Task entropy ----------
    H_task = -np.sum(
        np.clip(gamma_T, eps, 1.0) * np.log(np.clip(gamma_T, eps, 1.0)),
        axis=1
    )  # [N]

    # ---------- 2. Segment entropy ----------
    H_seg_matrix = -np.sum(
        np.clip(tau_S, eps, 1.0) * np.log(np.clip(tau_S, eps, 1.0)),
        axis=2
    )  # [N, T_seg]
    H_seg_mean = H_seg_matrix.mean(axis=1)  # [N]

    # ---------- 3. MQE entropy (only relevant MQEs) ----------
    # relevant MQEs = used in at least one segment
    relevant_mqe_mask = seg_mqe_mask.any(axis=0)  # [M]
    q_rel = q_mqe_any[:, relevant_mqe_mask]       # [N, M_relevant]

    p = np.clip(q_rel, eps, 1.0 - eps)
    H_mqe_matrix = - (p * np.log(p) + (1 - p) * np.log(1 - p))  # [N, M_relevant]
    H_mqe_mean = H_mqe_matrix.mean(axis=1)  # [N]

    # ---------- 4. Composite mechanistic ----------
    H_mech = 0.5 * (H_seg_mean + H_mqe_mean)  # [N]

    summary = {
        "H_task_mean": H_task.mean(),
        "H_task_std":  H_task.std(),
        "H_seg_mean":  H_seg_mean.mean(),
        "H_seg_std":   H_seg_mean.std(),
        "H_mqe_mean":  H_mqe_mean.mean(),
        "H_mqe_std":   H_mqe_mean.std(),
        "H_mech_mean": H_mech.mean(),
        "H_mech_std":  H_mech.std(),
    }

    if verbose:
        print("=== Decomposition / Mechanistic Uncertainty (relevant MQEs only) ===")
        print(f"Mean task entropy (H_T):    {summary['H_task_mean']:.6f} ± {summary['H_task_std']:.6f}")
        print(f"Mean segment entropy (H_S): {summary['H_seg_mean']:.6f} ± {summary['H_seg_std']:.6f}")
        print(f"Mean MQE entropy (H_M):     {summary['H_mqe_mean']:.6f} ± {summary['H_mqe_std']:.6f}")
        print(f"Composite mechanistic (H_mech): {summary['H_mech_mean']:.6f} ± {summary['H_mech_std']:.6f}")

    details = {
        "H_task": H_task,
        "H_seg": H_seg_mean,
        "H_mqe": H_mqe_mean,
        "H_mech": H_mech,
        "H_seg_matrix": H_seg_matrix,
        "H_mqe_matrix": H_mqe_matrix,
        "relevant_mqe_mask": relevant_mqe_mask,
    }

    return summary, details

import numpy as np

def compute_P_T_given_M(gamma_T, q_mqe_any, seg_mqe_mask, mqe_names):
    """
    Compute P(T = k | MQE_m = 1) for each MQE m and score k ∈ {0,1,2,3}.

    Inputs
    ------
      gamma_T      : [N, K]    posterior over task scores (full T→S→MQE model)
      q_mqe_any    : [N, M]    posterior P(MQE_m=1 in any segment | data)
      seg_mqe_mask : [T_seg,M] bool, True if MQE m is relevant in any segment t
      mqe_names    : list[str] length M, names of MQEs

    Returns
    -------
      mqe_names_rel   : list[str] for relevant MQEs only
      P_T_given_M_rel : [M_rel, K]  with rows m, columns k = 0..3
      P_M1_rel        : [M_rel]     marginal P(MQE_m=1)
    """
    eps = 1e-12
    N, K = gamma_T.shape
    M = q_mqe_any.shape[1]

    # 1) choose relevant MQEs (mapped to at least one segment)
    relevant_mqe_mask = seg_mqe_mask.any(axis=0)   # [M]
    idx_rel = np.where(relevant_mqe_mask)[0]
    mqe_names_rel = [mqe_names[j] for j in idx_rel]

    q_rel = q_mqe_any[:, idx_rel]  # [N, M_rel]
    M_rel = q_rel.shape[1]

    # 2) marginal P(T=k)
    P_T = gamma_T.mean(axis=0)     # [K]

    # 3) estimate P(M_m=1 | T=k) using posteriors
    #    numerator_mk = E[1{T=k} * 1{M_m=1}] ≈ (1/N) * Σ_i gamma_T[i,k] * q_rel[i,m]
    numerator = np.zeros((M_rel, K))
    for k in range(K):
        gamma_k = gamma_T[:, k][:, None]   # [N,1]
        numerator[:, k] = (gamma_k * q_rel).mean(axis=0)  # [M_rel]

    # 4) marginal P(M_m=1)
    P_M1_rel = q_rel.mean(axis=0)  # [M_rel]

    # 5) Bayes: P(T=k | M_m=1) = P(M_m=1 | T=k)*P(T=k)/P(M_m=1)
    P_T_given_M_rel = np.zeros((M_rel, K))
    for m in range(M_rel):
        denom = P_M1_rel[m] + eps
        for k in range(K):
            P_M_given_T = numerator[m, k] / (P_T[k] + eps)  # P(M_m=1 | T=k)
            P_T_given_M_rel[m, k] = P_M_given_T * P_T[k] / denom

        # normalize across k to be safe
        s = P_T_given_M_rel[m].sum()
        if s > 0:
            P_T_given_M_rel[m] /= s

    return mqe_names_rel, P_T_given_M_rel, P_M1_rel


def plot_P_T_given_M(mqe_names_rel, P_T_given_M_rel, save_path=None):
    """
    Plot P(T = k | MQE_m = 1) for k = 0..3, with
    one colored line per MQE (no two MQEs share a color).

    Parameters
    ----------
    mqe_names_rel : list[str] of length M
        Names of the selected MQEs.
    P_T_given_M_rel : np.ndarray, shape [M, K]
        P(T = k | MQE_m impaired) for each MQE m and score k.
    save_path : str or Path, optional
        If provided, save the figure there (PDF/PNG); otherwise call plt.show().
    """
    M_rel, K = P_T_given_M_rel.shape
    scores = np.arange(K)

    # paper-style defaults (optional if you already set rcParams globally)
    plt.rcParams.update({
        "figure.dpi": 300,
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(4.8, 3.0), constrained_layout=True)

    # qualitative, colorblind-friendly colormap with many distinct colors
    cmap = plt.get_cmap("tab20")   # up to 20 distinct hues

    for m in range(M_rel):
        color = cmap(m % cmap.N)
        ax.plot(
            scores,
            P_T_given_M_rel[m, :],
            marker="o",
            markersize=4,
            linewidth=1.3,
            color=color,
            label=mqe_names_rel[m],
        )

    ax.set_xticks(scores)
    ax.set_xticklabels([str(k) for k in scores])
    ax.set_xlabel("Task score $k$")
    ax.set_ylabel(r"$p(T = k \mid MQE impaired)$")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Task-score likelihood given MQE impairment")

    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)

    # Legend outside the plot if many MQEs
    ncol = 1 if M_rel <= 8 else 2
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=False,
        ncol=ncol,
    )

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# --------------------------------------------------
# 1) Dataset-level task posterior (after HBM)
# --------------------------------------------------

def plot_dataset_task_posterior(gamma_T, ax=None, title="Task posterior (dataset-level)"):
    """
    gamma_T : [N, K]  posterior p(T=k | data_i) after HBM / full hierarchy.

    Plots the dataset-averaged posterior P(T=k) = mean_i gamma_T[i,k].
    """
    N, K = gamma_T.shape
    P_T = gamma_T.mean(axis=0)  # [K]

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))

    x = np.arange(K)
    ax.bar(x, P_T)
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in range(K)])
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Task score k")
    ax.set_ylabel("$P(T = k)$ (dataset average)")
    ax.set_title(title)
    ax.grid(True, axis='y', alpha=0.3)

    return ax

# --------------------------------------------------
# 2) Dataset-level segment posterior per segment
# --------------------------------------------------

def plot_dataset_segment_posterior(tau_S, segment_names=None,
                                   ax=None, title="Segment posteriors (dataset-level)"):
    """
    tau_S : [N, T_seg, K]  posterior p(S_t = k | data_i)

    Computes for each segment t:
        P(S_t = k) = mean_i tau_S[i, t, k]
    and plots a heatmap [score k x segment t].
    """
    N, T_seg, K = tau_S.shape

    if segment_names is None:
        segment_names = [f"Seg {t+1}" for t in range(T_seg)]

    # [T_seg, K]
    P_S = tau_S.mean(axis=0)  # average over N -> P(S_t=k)
    # transpose to [K, T_seg] for plotting (rows: k, columns: segments)
    mat = P_S.T

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    im = ax.imshow(mat, aspect='auto', origin='lower',
                   vmin=0.0, vmax=1.0, cmap='viridis')

    ax.set_xticks(np.arange(T_seg))
    ax.set_xticklabels(segment_names)
    ax.set_yticks(np.arange(K))
    ax.set_yticklabels([str(k) for k in range(K)])
    ax.set_xlabel("Segments")
    ax.set_ylabel("Score k")
    ax.set_title(title)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("$P(S_t = k)$ (dataset average)")

    return ax

# --------------------------------------------------
# 3) Dataset-level MQE posterior (probability impaired)
# --------------------------------------------------

def plot_dataset_mqe_posterior(q_mqe_any, seg_mqe_mask, mqe_names,
                               ax=None, title="MQE posteriors (dataset-level)"):
    """
    q_mqe_any   : [N, M]    posterior P(MQE_m = 1 in any segment | data_i)
    seg_mqe_mask: [T_seg,M] bool, True if MQE m is used in >=1 segment
    mqe_names   : list[str] length M

    Computes P(MQE_m = 1) = mean_i q_mqe_any[i,m] for relevant MQEs only.
    """
    N, M = q_mqe_any.shape

    # Only use MQEs that are relevant in at least one segment (drop THS, PP, etc.)
    relevant_mask = seg_mqe_mask.any(axis=0)   # [M]
    idx_rel = np.where(relevant_mask)[0]

    names_rel = [mqe_names[j] for j in idx_rel]
    P_M1_rel = q_mqe_any[:, idx_rel].mean(axis=0)  # [M_rel]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    x = np.arange(len(idx_rel))
    ax.bar(x, P_M1_rel)

    ax.set_xticks(x)
    ax.set_xticklabels(names_rel, rotation=45, ha='right')
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("$P(\\text{MQE}_m = 1)$ (dataset average)")
    ax.set_xlabel("Movement Quality Elements (relevant only)")
    ax.set_title(title)
    ax.grid(True, axis='y', alpha=0.3)

    return ax

# --------------------------------------------------
# 4) Overall task posterior: pre- vs post-HBM
# --------------------------------------------------

def plot_overall_task_posterior_pre_post(gamma_T_hbm, gamma_T_raw=None,
                                         ax=None,
                                         title="Overall task posterior (pre vs post HBM)"):
    """
    gamma_T_hbm : [N, K]  posteriors after HBM / full hierarchy
    gamma_T_raw : [N, K]  (optional) pre-HBM or simpler model posteriors

    Plots dataset-averaged P(T=k) for HBM and, if provided, for the raw model.
    """
    N, K = gamma_T_hbm.shape
    P_T_hbm = gamma_T_hbm.mean(axis=0)  # [K]

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))

    x = np.arange(K)
    width = 0.35

    ax.bar(x, P_T_hbm, width=width, label="HBM / full hierarchy", alpha=0.8)

    if gamma_T_raw is not None:
        P_T_raw = gamma_T_raw.mean(axis=0)
        ax.bar(x + width, P_T_raw, width=width, label="Pre-HBM", alpha=0.6)

    shift = 0 if gamma_T_raw is None else width / 2
    ax.set_xticks(x + shift)
    ax.set_xticklabels([str(k) for k in range(K)])

    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Task score k")
    ax.set_ylabel("$P(T = k)$ (dataset average)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    return ax

def plot_P_M_given_T(mqe_names_rel, P_M_given_T, save_path=None):
    """
    Plot P(MQE_m = 1 | T = k) with:
      - x-axis: task scores k = 0..K-1
      - one line per MQE (each MQE has its own color, no reuse)

    Parameters
    ----------
    mqe_names_rel : list[str] of length M
        Names of the selected MQEs.
    P_M_given_T : np.ndarray, shape [M, K]
        P(MQE_m = 1 | T = k) for each MQE m and score k.
    save_path : str or Path, optional
        If provided, save the figure there (e.g., PDF/PNG); otherwise call plt.show().
    """
    M_rel, K = P_M_given_T.shape
    scores = np.arange(K)

    # Paper-style defaults (skip if you've set rcParams globally elsewhere)
    plt.rcParams.update({
        "figure.dpi": 300,
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(4.8, 3.0), constrained_layout=True)

    # Qualitative, colorblind-friendly colormap
    cmap = plt.get_cmap("tab20")  # up to 20 distinct colors

    for m in range(M_rel):
        color = cmap(m % cmap.N)
        ax.plot(
            scores,
            P_M_given_T[m, :],
            marker="o",
            markersize=4,
            linewidth=1.3,
            color=color,
            label=mqe_names_rel[m],
        )

    ax.set_xticks(scores)
    ax.set_xticklabels([str(k) for k in scores])
    ax.set_ylim(0.0, 1.0)

    ax.set_xlabel("Task score $k$")
    ax.set_ylabel(r"$p(MQE impaired \mid T = k)$")
    ax.set_title("Likelihood of MQE impairment given task score")

    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)

    # Legend outside to avoid overlapping with lines
    ncol = 1 if M_rel <= 8 else 2
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=False,
        ncol=ncol,
    )

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()



def compute_P_M_given_T(gamma_T, q_mqe_any, seg_mqe_mask, mqe_names):
    """
    Compute P(MQE_m = 1 | T = k) for each MQE m and task score k.

    gamma_T      : [N, K]   task posteriors (e.g., gamma_T_full)
    q_mqe_any    : [N, M]   P(MQE_m = 1 in any segment | data)
    seg_mqe_mask : [T_seg,M] bool, True if MQE m is used in >=1 segment
    mqe_names    : list[str] length M

    Returns:
      mqe_names_rel : list[str]      names of relevant MQEs (e.g., 12)
      P_M_given_T   : [M_rel, K]     rows = MQEs, cols = scores 0..3
    """
    eps = 1e-12
    N, K = gamma_T.shape
    N2, M = q_mqe_any.shape
    assert N == N2

    # Use only MQEs that belong to at least one segment (drop THS, PP, etc.)
    relevant_mask = seg_mqe_mask.any(axis=0)   # [M]
    idx_rel = np.where(relevant_mask)[0]
    mqe_names_rel = [mqe_names[j] for j in idx_rel]
    M_rel = len(idx_rel)

    # Posterior-weighted counts
    # num[m,k] ≈ Σ_i gamma_T[i,k] * q_mqe_any[i,m]
    num = np.zeros((M_rel, K))
    for j_rel, m in enumerate(idx_rel):
        q_m = q_mqe_any[:, m]                 # [N]
        num[j_rel, :] = (gamma_T * q_m[:, None]).sum(axis=0)

    # Denominator Σ_i gamma_T[i,k] (effective number of tasks with label k)
    gamma_sum = gamma_T.sum(axis=0)           # [K]

    # P(M_m = 1 | T = k) = num[m,k] / gamma_sum[k]
    P_M_given_T = num / (gamma_sum[None, :] + eps)

    return mqe_names_rel, P_M_given_T


def plot_segment_posterior_lines(P_S, segment_names=None, title="Segment posterior"):
    """
    Line plot of P(S_t = k) across segments.

    P_S           : [T_seg, K]  array with P(S_t = k) averaged over dataset
                    (rows: segments, cols: scores 0..K-1)
    segment_names : list of length T_seg (optional)
    title         : plot title
    """
    T_seg, K = P_S.shape
    x = np.arange(T_seg)

    if segment_names is None:
        segment_names = [f"Seg {t+1}" for t in range(T_seg)]

    plt.figure(figsize=(6, 4))

    for k in range(K):
        plt.plot(
            x,
            P_S[:, k],
            marker="o",
            linewidth=1.5,
            label=f"Score {k}"
        )

    plt.xticks(x, segment_names)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Segments")
    plt.ylabel("P(Segment score = k)")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    """
    Main script: run all label models and generate paper-quality figures.
    """

    import matplotlib.pyplot as plt
    from pathlib import Path

    # ------------------------------------------------------------------
    # Global plotting style for paper
    # ------------------------------------------------------------------
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # Colorblind-friendly palette
    cb_blue   = "#4C72B0"
    cb_orange = "#DD8452"
    cb_green  = "#55A868"
    cb_red    = "#C44E52"
    cb_purple = "#8172B3"
    cb_gray   = "#999999"

    # Where to save figures
    csv_dir = r"D:\nature_everything"
    fig_dir = Path(csv_dir) / "figures_tochi"
    fig_dir.mkdir(exist_ok=True, parents=True)

    # ------------------------------------------------------------------
    # Load ratings and build R_task / R_seg as before
    # ------------------------------------------------------------------
    np.random.seed(0)
    T_seg = 4
    C = 2
    M_mqe = 10
    n_task_types = 15
    K = 4  # ARAT scores 0..3

    records = load_video_segments_info(csv_dir)
    task_ratings_dict, segment_ratings_dict, composite_rating_dict = load_rating_info(csv_dir)

    N_instances = len(task_ratings_dict)
    task_type_id = np.random.randint(0, n_task_types, size=N_instances)

    clinician_keys = ["t1", "t2"]
    C = len(clinician_keys)

    instance_ids = sorted(task_ratings_dict.keys())
    N_instances = len(instance_ids)

    R_task = np.full((N_instances, C), -1, dtype=int)
    idx_to_instance_id = instance_ids
    instance_id_to_idx = {inst_id: i for i, inst_id in enumerate(instance_ids)}

    for i, inst_id in enumerate(instance_ids):
        rating_dict = task_ratings_dict[inst_id]
        for c, ck in enumerate(clinician_keys):
            if ck in rating_dict and rating_dict[ck] is not None:
                R_task[i, c] = int(rating_dict[ck])

    print("R_task shape:", R_task.shape)
    print("First few rows of R_task:\n", R_task[:5])
    print("First few instance ids:", idx_to_instance_id[:5])

    # ---------------------- segment ratings ---------------------------
    valid_segments = {1, 2, 3, 4}
    T_seg = 4
    seg_index = {1: 0, 2: 1, 3: 2, 4: 3}

    R_seg = np.full((N_instances, T_seg, C), -1, dtype=int)

    for i, inst_id in enumerate(idx_to_instance_id):
        rating_entry = segment_ratings_dict.get(inst_id, None)
        if rating_entry is None:
            continue

        all_seg_ids = set()
        for ck in clinician_keys:
            seg_dict = rating_entry.get(ck, {})
            for s in seg_dict.keys():
                all_seg_ids.add(int(s))

        if not all(s in valid_segments for s in all_seg_ids):
            continue

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

    # ------------------------------------------------------------------
    # PRE-HBM MODELS: task-only and task+segment
    # ------------------------------------------------------------------
    post_task, pi_task, Theta_task = ds_multiclass(R_task, K=K, max_iter=1000, verbose=True)
    H_task = entropy(post_task)           # [N]
    T_hat = post_task.argmax(axis=1)

    gamma_T, q_S, pi_ts, Theta_task_ts, Theta_seg_ts, Phi_t = em_task_segment(
        R_task, R_seg, K=K, max_iter=1000, verbose=True
    )
    H_task_ts = entropy(gamma_T)
    P_T=post_task.mean(axis=0)
    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    x = np.arange(K)
    ax.bar(x, P_T, color=[cb_blue, cb_green, cb_red, cb_purple],
           edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xlabel(r"Task score $k$")
    ax.set_ylabel(r"$p(T = k)$ (dataset average)")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Task posterior (after intervention, pre-HBM)")
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(fig_dir / "dtask_posterior_after_intervention_prehbm.png", dpi=300, bbox_inches='tight')

    # Histogram: entropy task-only vs task+segment
    fig, ax = plt.subplots(figsize=(3.3, 2.7))
    bins = 30
    ax.hist(H_task, bins=bins, alpha=0.5, label="Task-only", color=cb_blue, edgecolor="black", linewidth=0.4)
    ax.hist(H_task_ts, bins=bins, alpha=0.5, label="Task + segments", color=cb_orange, edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Task entropy H(T)")
    ax.set_ylabel("Number of tasks")
    #ax.set_title("Entropy distribution: task-only vs task+segment")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(fig_dir / "entropy_taskonly_vs_taskseg.png", dpi=300, bbox_inches='tight')

    # ΔH histogram: task-only -> task+segment
    delta_H = H_task - H_task_ts  # >0 means uncertainty reduced
    fig, ax = plt.subplots(figsize=(3.3, 2.7))
    ax.hist(delta_H, bins=30, color=cb_blue, alpha=0.8, edgecolor="black", linewidth=0.4)
    ax.axvline(0.0, color=cb_gray, linestyle="--", linewidth=1.0)
    ax.set_xlabel("ΔH = H_task-only − H_task+seg")
    ax.set_ylabel("Number of tasks")
    #ax.set_title("Per-task entropy reduction from adding segments")
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(fig_dir / "deltaH_taskonly_to_taskseg.png", dpi=300, bbox_inches='tight')


    # Scatter: H_task vs H_task_ts
    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    ax.scatter(H_task, H_task_ts, s=8, color=cb_blue, alpha=0.7)
    min_val = min(H_task.min(), H_task_ts.min())
    max_val = max(H_task.max(), H_task_ts.max())
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color=cb_gray, linewidth=1.0)
    ax.set_xlabel("Entropy (task-only)")
    ax.set_ylabel("Entropy (task+segments)")
    ax.set_title("Per-task entropy: task-only vs task+segments")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(fig_dir / "scatter_entropy_task_vs_taskseg.png", dpi=300, bbox_inches='tight')

    # Hard subset histogram
    hard_mask = H_task > 0.8
    print("Number of 'hard' tasks:", hard_mask.sum())
    fig, ax = plt.subplots(figsize=(3.3, 2.7))
    ax.hist(H_task[hard_mask], bins=20, alpha=0.5, label="Task-only (hard subset)",
            color=cb_blue, edgecolor="black", linewidth=0.4)
    ax.hist(H_task_ts[hard_mask], bins=20, alpha=0.5, label="Task+segments (hard subset)",
            color=cb_orange, edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Task entropy H(T) (hard subset)")
    ax.set_ylabel("Number of tasks")
    ax.set_title("Effect of segments on high-uncertainty tasks")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(fig_dir / "entropy_hard_subset_task_vs_taskseg.png", dpi=300, bbox_inches='tight')

    # ------------------------------------------------------------------
    # Task + MQE and Task + seg + MQE (flat)
    # ------------------------------------------------------------------
    X_mqe_task, X_mqe_seg = build_mqe_arrays(
        composite_rating_dict,
        idx_to_instance_id,
        T_seg=4,
        clinician_keys=("t1", "t2"),
        M_mqe=None,
    )

    gamma_T_mqe, pi_mqe, Theta_task_mqe, Beta_task = em_task_mqe(
        R_task, X_mqe_task, K=K, max_iter=1000, verbose=True
    )
    H_task_mqe = entropy(gamma_T_mqe)

    gamma_T_seg_mqe, pi_seg_mqe, Theta_task_seg_mqe, Theta_seg_seg_mqe, Beta_seg = em_task_seg_mqe(
        R_task, R_seg, X_mqe_seg, K=K, max_iter=1000, verbose=True
    )
    H_task_seg_mqe = entropy(gamma_T_seg_mqe)


    # Overlaid entropy distributions (3 flat models)
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    bins = 30
    ax.hist(H_task,          bins=bins, alpha=0.45, label="Task-only",
            color=cb_blue, edgecolor="black", linewidth=0.3)
    ax.hist(H_task_mqe,      bins=bins, alpha=0.45, label="Task + MQE",
            color=cb_green, edgecolor="black", linewidth=0.3)
    ax.hist(H_task_seg_mqe,  bins=bins, alpha=0.45, label="Task + seg + MQE",
            color=cb_orange, edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Task entropy H(T)")
    ax.set_ylabel("Number of tasks")
    ax.set_title("Entropy distributions across flat label models")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(fig_dir / "entropy_flat_models_comparison.png", dpi=300, bbox_inches='tight')


    # ΔH histograms: task→MQE and MQE→seg+MQE
    delta_H_task_to_mqe = H_task - H_task_mqe
    delta_H_mqe_to_seg  = H_task_mqe - H_task_seg_mqe

    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    ax.hist(
        delta_H_task_to_mqe,
        bins=30,
        alpha=0.7,
        label=r"$\Delta H$ (task-only $\rightarrow$ task+MQE)",
        color=cb_blue,
        edgecolor="black",
        linewidth=0.3,
    )
    ax.hist(
        delta_H_mqe_to_seg,
        bins=30,
        alpha=0.7,
        label=r"$\Delta H$ (task+MQE $\rightarrow$ task+seg+MQE)",
        color=cb_orange,
        edgecolor="black",
        linewidth=0.3,
    )
    ax.legend(frameon=False, loc='best',fontsize=5)
    ax.axvline(0.0, color=cb_gray, linestyle="--", linewidth=1.0)
    ax.set_xlabel(r"Entropy change $\Delta H$")
    ax.set_ylabel("Number of tasks")
    ax.set_title("Entropy reduction per task")
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(fig_dir / "deltaH_taskonly_to_mqe_and_mqe_to_seg_mqe.png", dpi=300, bbox_inches='tight')



    # ------------------------------------------------------------------
    # Live pre-intervention DS model (1 clinician)
    # ------------------------------------------------------------------
    LIVE_CSV = Path(csv_dir) / "live_rating_cleaned.csv"
    df_live = pd.read_csv(LIVE_CSV)

    if "Rating" in df_live.columns:
        rating_col = "Rating"
    else:
        candidates = [c for c in df_live.columns if "rating" in c.lower()]
        if not candidates:
            raise ValueError(f"Could not find a Rating column in {df_live.columns}")
        rating_col = candidates[0]

    df_live[rating_col] = df_live[rating_col].astype(int)
    ratings_live = df_live[rating_col].to_numpy()
    N_live = ratings_live.shape[0]
    R_task_live = ratings_live.reshape(N_live, 1)


    post_task_live, pi_live, Theta_live = ds_multiclass(
        R_task_live, K=K, max_iter=500, verbose=True
    )

    def entropy_np(p, axis=-1, eps=1e-12):
        p_clip = np.clip(p, eps, 1.0)
        return -np.sum(p_clip * np.log(p_clip), axis=axis)

    H_task_live = entropy_np(post_task_live)

    # dataset-level posterior (live)
    P_T_live = post_task_live.mean(axis=0)

    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    x = np.arange(K)
    ax.bar(x, P_T_live, color=[cb_blue, cb_green, cb_red, cb_purple],
           edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xlabel(r"Task score $k$")
    ax.set_ylabel(r"$p(T = k)$ (dataset average)")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Task posterior (live pre-intervention, 1 clinician)")
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(fig_dir / "dtask_posterior_live_preintervention.png", dpi=300, bbox_inches='tight')

    # entropy distribution for live ratings
    fig, ax = plt.subplots(figsize=(3.3, 2.7))
    ax.hist(H_task_live, bins=30, alpha=0.8, color=cb_blue,
            edgecolor="black", linewidth=0.4)
    ax.set_xlabel(r"Task entropy $H(T)$")
    ax.set_ylabel("Number of tasks")
    ax.set_title("Uncertainty of task scores (live pre-intervention)")
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(fig_dir / "entropy_live_preintervention.png", dpi=300, bbox_inches='tight')


    # Compare offline DS vs live pre-intervention
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    bins = 30
    ax.hist(H_task,      bins=bins, alpha=0.5, label="Offline DS (recorded)",
            color=cb_blue, edgecolor="black", linewidth=0.3)
    ax.hist(H_task_live, bins=bins, alpha=0.5, label="Live pre-intervention",
            color=cb_orange, edgecolor="black", linewidth=0.3)
    ax.set_xlabel(r"Task entropy $H(T)$")
    ax.set_ylabel("Number of tasks")
    ax.set_title("Entropy comparison: offline vs live pre-intervention")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(fig_dir / "entropy_offline_vs_live.png", dpi=300, bbox_inches='tight')


    # ------------------------------------------------------------------
    # FULL HBM: T→S→MQE
    # ------------------------------------------------------------------
    Y_mqe = build_mqe_rater_arrays(
        composite_rating_dict,
        idx_to_instance_id,
        T_seg=4,
        clinician_keys=("t1", "t2"),
        M_mqe=None,
    )

    gamma_T_full, tau_S_full, pi_full, Theta_task_full, Theta_seg_full, Phi_t_full, Beta_mqe_full = \
        em_full_TS_MQE_meanfield(R_task, R_seg, Y_mqe, K=K, max_iter=1000, verbose=True)

    H_task_full = entropy(gamma_T_full)
    H_seg_full = entropy(tau_S_full)           # [N, T_seg]
    H_seg_mean = H_seg_full.mean(axis=1)
    # After em_full_TS_MQE_meanfield()
    q_mqe_seg, q_mqe_any, seg_mqe_mask = compute_mqe_posterior_probs(tau_S_full, Beta_mqe_full)

    def bernoulli_entropy(p, eps=1e-12):
        p = np.clip(p, eps, 1.0 - eps)
        return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))

    # H_Y(i,t,m)
    H_mqe = bernoulli_entropy(q_mqe_seg)            # [N, T_seg, M]

    # per-task mean MQE entropy \bar{H}_Y(i)
    H_mqe_mean_per_task = H_mqe.mean(axis=(1, 2))   # [N]


    # Task entropy distributions: task-only vs full hierarchy
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    bins = 30
    ax.hist(H_task,      bins=bins, alpha=0.45, label="Task-only",
            color=cb_blue, edgecolor="black", linewidth=0.3)
    ax.hist(H_task_full, bins=bins, alpha=0.45, label="Full T→S→MQE",
            color=cb_orange, edgecolor="black", linewidth=0.3)
    ax.set_xlabel(r"Task entropy $H(T)$")
    ax.set_ylabel("Number of tasks")
    ax.set_title("Task entropy: task-only vs full hierarchy")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    fig.tight_layout()

    fig.savefig(fig_dir / "entropy_taskonly_vs_full_hbm.png", dpi=300, bbox_inches='tight')

    # ΔH: flat seg+MQE vs full hierarchy
    delta_full_vs_flat_seg = H_task_seg_mqe - H_task_full
    fig, ax = plt.subplots(figsize=(3.3, 2.7))
    ax.hist(delta_full_vs_flat_seg, bins=30, alpha=0.8, color=cb_blue,
            edgecolor="black", linewidth=0.3)
    ax.axvline(0.0, color=cb_gray, linestyle="--", linewidth=1.0)
    ax.set_xlabel("ΔH = H_flat(seg+MQE) − H_full(T→S→MQE)")
    ax.set_ylabel("Number of tasks")
    ax.set_title("Entropy reduction from full hierarchical model")
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(fig_dir / "deltaH_flatsegmqe_to_fullhbm.png", dpi=300, bbox_inches='tight')

    delta_full_vs_task_only = H_task - H_task_full
    fig, ax = plt.subplots(figsize=(3.3, 2.7))
    ax.hist(delta_full_vs_task_only, bins=30, alpha=0.8, color=cb_green,
            edgecolor="black", linewidth=0.3)
    ax.axvline(0.0, color=cb_gray, linestyle="--", linewidth=1.0)
    ax.set_xlabel("ΔH = H_task-only − H_full(T→S→MQE)")
    ax.set_ylabel("Number of tasks")
    ax.set_title("Entropy reduction vs task-only model")
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(fig_dir / "deltaH_taskonly_to_fullhbm.png", dpi=300, bbox_inches='tight')


    # Scatter: task vs mean segment entropy under full hierarchy
    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    ax.scatter(H_task_full, H_seg_mean, s=8, color=cb_blue, alpha=0.7)
    ax.set_xlabel(r"Task entropy $H_T$ (full T→S→MQE)")
    ax.set_ylabel(r"Mean segment entropy per task $\bar{H}_S$")
    ax.set_title("Relation between task- and segment-level uncertainty")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(fig_dir / "scatter_task_vs_segment_entropy_fullhbm.png", dpi=300, bbox_inches='tight')


    # Dataset-level task posterior *after* HBM
    P_T_full = gamma_T_full.mean(axis=0)    # shape [K]
    P_T_full = P_T_full / P_T_full.sum()    # optional, just to be safe

    print(P_T_full, P_T_full.sum())         # should sum ~ 1.0

    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    x = np.arange(K)
    ax.bar(x, P_T_full, color=[cb_blue, cb_green, cb_red, cb_purple],
        edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xlabel(r"Task score $k$")
    ax.set_ylabel(r"$p(T = k)$ (dataset average)")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Task posterior (after intervention, post-HBM)")
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(fig_dir / "dtask_posterior_after_intervention_posthbm.png",
                dpi=300, bbox_inches='tight')

    # ---------- Post-HBM: task/flat → full T→S→MQE ----------
    delta_H_task_to_full = H_task - H_task_full
    delta_H_flat_to_full = H_task_seg_mqe - H_task_full  # flat task+seg+MQE → full hierarchy

    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    ax.hist(
        delta_H_task_to_full,
        bins=30,
        alpha=0.7,
        label=r"$\Delta H$ (task-only $\rightarrow$ task+MQE)",
        color=cb_blue,
        edgecolor="black",
        linewidth=0.3,
    )
    ax.hist(
        delta_H_flat_to_full,
        bins=30,
        alpha=0.7,
        label=r"$\Delta H$ (task+MQE $\rightarrow$ task+seg+MQE)",
        color=cb_orange,
        edgecolor="black",
        linewidth=0.3,
    )
    ax.legend(frameon=False, loc='best')
    ax.axvline(0.0, color=cb_gray, linestyle="--", linewidth=1.0)
    ax.set_xlabel(r"Entropy change $\Delta H$")
    ax.set_ylabel("Number of tasks")
    ax.set_title("Entropy reduction per task (full hierarchy)")
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(
        fig_dir / "deltaH_taskonly_and_flat_to_fullTSMQE.png",
        dpi=300,
        bbox_inches="tight",
    )
    # (Optional) show all figures interactively if running in a notebook/terminal
    plt.show()

    # full dataset, no masking
    mqe_names_rel, P_T_given_M_rel, P_M1_rel = compute_P_T_given_M(
        gamma_T_full,   # [N, K] all trials
        q_mqe_any,         # [N, M] all trials
        seg_mqe_mask,      # [T_seg, M]
        MQE_NAMES          # full list of MQE names
    )
    mqe_names_rel, P_M_given_T = compute_P_M_given_T(
        gamma_T_full,
        q_mqe_any,
        seg_mqe_mask,
        MQE_NAMES
    )

    plot_P_T_given_M(mqe_names_rel, P_T_given_M_rel,"D:\\nature_everything\\figures_tochi\\PT_given_M.png")
    plot_P_M_given_T(mqe_names_rel, P_M_given_T,"D:\\nature_everything\\figures_tochi\\PM_given_T.png")

    # ---------- 3. MQE entropy (mean per task) H_M ----------
    # use only MQEs that are relevant for at least one segment
    eps = 1e-12
    relevant_mqe_mask = seg_mqe_mask.any(axis=0)   # [M]
    q_rel = q_mqe_any[:, relevant_mqe_mask]        # [N, M_relevant]

    p = np.clip(q_rel, eps, 1.0 - eps)
    H_mqe_matrix = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))  # [N, M_rel]
    H_mqe_mean = H_mqe_matrix.mean(axis=1)  # [N]

    # ---------- Scatter: segment vs MQE ----------
    fig, ax = plt.subplots(figsize=(3.0, 3.0))

    ax.scatter(
        H_seg_mean,
        H_mqe_mean,
        s=12,
        alpha=0.6,
        linewidths=0
    )

    # Reference line y = x
    min_val = min(H_seg_mean.min(), H_mqe_mean.min())
    max_val = max(H_seg_mean.max(), H_mqe_mean.max())
    ax.plot([min_val, max_val], [min_val, max_val],
            linestyle="--", linewidth=0.8)

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    ax.set_xlabel(r"Mean segment entropy per task $\bar{H}_S$")
    ax.set_ylabel(r"Mean MQE entropy per task $\bar{H}_M$")
    ax.set_title("Relation between segment- and MQE-level uncertainty")

    ax.grid(True, axis="both", linestyle=":", linewidth=0.5, alpha=0.6)
    fig.tight_layout()

    fig.savefig(
        fig_dir / "dsegment_vs_mqe_uncertainty.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()


    print("Offline DS mean H(T):", H_task.mean())
    print("Mean task entropy (task+segment model):", H_task_ts.mean())
    print("Mean entropy (task+MQE):         ", H_task_mqe.mean())
    print("Mean entropy (task+seg+MQE):     ", H_task_seg_mqe.mean())
    print("Fraction of tasks with ΔH > 0:", (delta_H > 0).mean())
    print("Frac with ΔH(task→MQE) > 0:", (delta_H_task_to_mqe > 0).mean())
    print("Frac with ΔH(MQE→seg+MQE) > 0:", (delta_H_mqe_to_seg > 0).mean())
    print("Mean task entropy (live pre-intervention):", H_task_live.mean())
    print("Std  task entropy (live pre-intervention):", H_task_live.std())    
    print("Mean entropy Task(full T→S→MQE):       ", H_task_full.mean())
    print("Mean entropy Segment(full T→S→MQE):       ", H_seg_full.mean(axis=1))
    print("Mean entropy MQE(full T→S→MQE):       ", H_mqe_mean_per_task.mean())
    print("Fraction of tasks with entropy reduced vs flat seg+MQE:",(delta_full_vs_flat_seg > 0).mean())
    print("Mean H_M:", H_mqe_mean.mean())

    


    # ---------- TABLE 1: task-level entropy summary with fraction of max ----------

    # Max entropy for 4-point task scores (0–3), assuming natural log
    H_max_task = np.log(4.0)

    task_models = [
        ("Live pre-intervention (1 clinician, single camera)", H_task_live),
        ("Recorded: task-only DS",                            H_task),
        ("Recorded: task+segment (flat)",                     H_task_ts),
        ("Recorded: task+seg+MQE (flat)",                     H_task_seg_mqe),
        ("Recorded: full T→S→MQE (HBM)",                      H_task_full),
    ]

    rows1 = []
    for name, arr in task_models:
        mean_H = float(arr.mean())
        std_H  = float(arr.std())
        frac   = mean_H / H_max_task
        rows1.append({
            "Model / stage": name,
            "Mean H(T) [nats]": mean_H,
            "Std H(T) [nats]": std_H,
            "Fraction of max entropy": frac,
        })

    table1 = pd.DataFrame(rows1).set_index("Model / stage")

    # Sanity check print
    print(table1.to_string(float_format=lambda x: f"{x:.4f}"))

    # LaTeX for the paper
    latex1 = table1.to_latex(
        float_format="%.4f",
        escape=False,
        caption="Task-level entropy $H(T)$ across models and design stages. Fraction of max entropy uses $H_{\max} = \\log 4$ for the 4-point ARAT scale.",
        label="tab:task_entropy"
    )
    print("\nLaTeX for Table 1:\n")
    print(latex1)

    # Optional: save as CSV
    table1.to_csv("D:/nature_everything/figures_tochi/table_task_entropy.csv", float_format="%.6f")




    # ---------- TABLE 2: hierarchical decomposition under full HBM ----------

    # per-task mean segment entropy from H_seg_full [N, T_seg]
    H_seg_mean_per_task = H_seg_full.mean(axis=1)   # [N]

    max_task_seg = np.log(4.0)  # max entropy for 4-point scores
    max_mqe      = np.log(2.0)  # max entropy for binary MQEs

    rows2 = [
        {
            "Level": "Task entropy $H_T$",
            "Mean [nats]": float(H_task_full.mean()),
            "Std [nats]": float(H_task_full.std()),
            "Fraction of max": float(H_task_full.mean() / max_task_seg),
        },
        {
            "Level": "Mean segment entropy $\\bar{H}_S$",
            "Mean [nats]": float(H_seg_mean_per_task.mean()),
            "Std [nats]": float(H_seg_mean_per_task.std()),
            "Fraction of max": float(H_seg_mean_per_task.mean() / max_task_seg),
        },
        {
            "Level": "Mean MQE entropy $\\bar{H}_Y$",
            "Mean [nats]": float(H_mqe_mean_per_task.mean()),
            "Std [nats]": float(H_mqe_mean_per_task.std()),
            "Fraction of max": float(H_mqe_mean_per_task.mean() / max_mqe),
        },
        {
            "Level": "Mean MQE entropy $\\bar{H}_M$",
            "Mean [nats]": float(H_mqe_mean.mean()),
            "Std [nats]": float(H_mqe_mean.std()),
            "Fraction of max": float(H_mqe_mean.mean() / max_mqe),
        },        
    ]

    table2 = pd.DataFrame(rows2).set_index("Level")

    # Pretty print
    print(table2.to_string(float_format=lambda x: f"{x:.4f}"))

    # LaTeX for the paper
    latex2 = table2.to_latex(
        float_format="%.4f",
        escape=False,
        caption="Hierarchical decomposition of entropy under the full $T\\rightarrow S\\rightarrow$MQE model.",
        label="tab:hier_entropy"
    )
    print("\nLaTeX for Table 2:\n")
    print(latex2)

    # Optional: save as CSV
    table2.to_csv("D:/nature_everything/figures_tochi/table_hier_entropy.csv", float_format="%.6f")









    ###########################full HBM model done#######################################


    # seg_entropy, mean_seg_entropy = compute_segment_entropy(tau_S_full)
    # print("Mean segment entropy (full model):", mean_seg_entropy.mean())

    q_mqe_seg, q_mqe_any, seg_mqe_mask = compute_mqe_posterior_probs(
        tau_S_full,
        Beta_mqe_full
    )


    # # e.g. mean probability each MQE is impaired (across tasks)
    # mean_mqe_prob = q_mqe_any.mean(axis=0)
    # for m, name in enumerate(MQE_NAMES):
    #     print(name, mean_mqe_prob[m])
    # # pick a trial whose MAP task is 2
    # idx_2 = np.where(gamma_T_full.argmax(axis=1) == 2)[0][0]
    # explain_trial(idx_2, gamma_T_full, tau_S_full,
    #             q_mqe_seg, q_mqe_any, MQE_NAMES)


    # idx_sel, cluster_ids, centers = cluster_mqe_patterns_for_label(
    #     gamma_T_full,
    #     q_mqe_any,
    #     target_label=2,
    #     min_conf=0.9,
    #     n_clusters=3,
    #     random_state=42
    # )

    # # Print cluster summaries
    # for cl in range(3):
    #     print(f"\n=== Type 2 subtype #{cl} ===")
    #     center = centers[cl]  # [M]
    #     order = np.argsort(-center)
    #     for m in order[:5]:
    #         print(f"  {MQE_NAMES[m]}: P_imp≈{center[m]:.3f}")

    # print(f"\nTotal '2' trials considered: {idx_sel.size}")


    # summary, details = summarize_mechanistic_uncertainty_full(
    #     gamma_T_full,
    #     tau_S_full,
    #     q_mqe_any,
    #     seg_mqe_mask
    # )

    eps = 1e-12

    # ---------- 1. Task entropy H_T ----------
    H_task_hbm = -np.sum(
        np.clip(gamma_T_full, eps, 1.0)
        * np.log(np.clip(gamma_T_full, eps, 1.0)),
        axis=1
    )  # [N]

    # ---------- 2. Segment entropy (mean per task) H_S ----------
    H_seg_matrix = -np.sum(
        np.clip(tau_S_full, eps, 1.0)
        * np.log(np.clip(tau_S_full, eps, 1.0)),
        axis=2
    )  # [N, T_seg]
    H_seg_mean = H_seg_matrix.mean(axis=1)  # [N]

    # ---------- 3. MQE entropy (mean per task) H_M ----------
    # use only MQEs that are relevant for at least one segment
    relevant_mqe_mask = seg_mqe_mask.any(axis=0)   # [M]
    q_rel = q_mqe_any[:, relevant_mqe_mask]        # [N, M_relevant]

    p = np.clip(q_rel, eps, 1.0 - eps)
    H_mqe_matrix = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))  # [N, M_relevant]
    H_mqe_mean = H_mqe_matrix.mean(axis=1)  # [N]

    print("Mean H_T:", H_task_hbm.mean())
    print("Mean H_S:", H_seg_mean.mean())
    print("Mean H_M:", H_mqe_mean.mean())

    # ---------- 4. Scatter: task vs MQE ----------
    plt.figure(figsize=(7, 5))
    plt.scatter(H_task_hbm, H_mqe_mean, s=10, alpha=0.6)
    plt.xlabel("Task entropy $H_T$")
    plt.ylabel("Mean MQE entropy per task $\\bar{H}_M$")
    plt.title("Relation between task- and MQE-level uncertainty")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ---------- 5. Scatter: segment vs MQE ----------
    plt.figure(figsize=(7, 5))
    plt.scatter(H_seg_mean, H_mqe_mean, s=10, alpha=0.6)
    # optional reference line y = x
    lims = [
        min(H_seg_mean.min(), H_mqe_mean.min()),
        max(H_seg_mean.max(), H_mqe_mean.max())
    ]
    plt.plot(lims, lims, "k--", linewidth=1, alpha=0.4)
    plt.xlim(lims)
    plt.ylim(lims)

    plt.xlabel("Mean segment entropy per task $\\bar{H}_S$")
    plt.ylabel("Mean MQE entropy per task $\\bar{H}_M$")
    plt.title("Relation between segment- and MQE-level uncertainty")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()




    # ============================================================
    # 1) Task posterior BEFORE HBM (task-only Dawid–Skene)
    #    Uses: post_task  [N, K] from ds_multiclass(...)
    # ============================================================

    P_T_task_only = post_task.mean(axis=0)  # [K]

    plt.figure(figsize=(4, 4))
    x = np.arange(P_T_task_only.shape[0])
    plt.bar(x, P_T_task_only)
    plt.xticks(x, [str(k) for k in range(P_T_task_only.shape[0])])
    plt.ylim(0.0, 1.0)
    plt.xlabel("Task score k")
    plt.ylabel("P(T = k) (dataset average)")
    plt.title("Task posterior before HBM (task-only)")
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ============================================================
    # 2) Segment posterior BEFORE HBM (from q_S)
    #    Uses: q_S [N, T_seg, K] from em_task_segment(...)
    # ============================================================

    P_S_before = q_S.mean(axis=0)   # [T_seg, K]
    segment_names = ["IP", "T", "MTR", "PR"] if P_S_before.shape[0] == 4 else None

    plot_segment_posterior_lines(
        P_S_before,
        segment_names=segment_names,
        title="Segment posterior before HBM (task+segment EM)"
    )



    # ============================================================
    # 3) MQE posterior BEFORE HBM (from flat task+seg+MQE EM)
    #    Uses: pi_seg_mqe [K], Beta_seg [K,M], seg_mqe_mask [T_seg,M], MQE_NAMES
    # ============================================================

    # Global P(MQE_m = 1) under the flat task+seg+MQE model:
    P_M1_flat = (pi_seg_mqe[:, None] * Beta_seg).sum(axis=0)  # [M]

    # Only MQEs that actually appear in at least one segment
    relevant_mask = seg_mqe_mask.any(axis=0)   # [M]
    idx_rel = np.where(relevant_mask)[0]

    mqe_names_rel = [MQE_NAMES[j] for j in idx_rel]
    P_M1_rel_flat = P_M1_flat[idx_rel]

    plt.figure(figsize=(8, 4))
    x = np.arange(len(idx_rel))
    plt.bar(x, P_M1_rel_flat)
    plt.xticks(x, mqe_names_rel, rotation=45, ha='right')
    plt.ylim(0.0, 1.0)
    plt.ylabel("P(MQE_m = 1) (dataset average)")
    plt.xlabel("Movement Quality Elements (relevant only)")
    plt.title("MQE posterior before HBM (dataset-level)")
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ============================================================
    # 4) Task posterior AFTER HBM (full T→S→MQE model)
    #    Uses: gamma_T_full [N, K] from em_full_TS_MQE_meanfield(...)
    # ============================================================

    P_T_after = gamma_T_full.mean(axis=0)  # [K]

    plt.figure(figsize=(4, 4))
    x = np.arange(P_T_after.shape[0])
    plt.bar(x, P_T_after)
    plt.xticks(x, [str(k) for k in range(P_T_after.shape[0])])
    plt.ylim(0.0, 1.0)
    plt.xlabel("Task score k")
    plt.ylabel("P(T = k) (dataset average)")
    plt.title("Task posterior after HBM (full T→S→MQE)")
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


    # ============================================================
    # 2') Segment posterior AFTER HBM (from tau_S_full)
    #      Uses: tau_S_full [N, T_seg, K]
    # ============================================================

    P_S_after = tau_S_full.mean(axis=0)   # [T_seg, K]

    segment_names = ["IP", "T", "MTR", "PR"] if P_S_after.shape[0] == 4 else None

    plot_segment_posterior_lines(
        P_S_after,
        segment_names=segment_names,
        title="Segment posterior after HBM (full T→S→MQE)"
    )


    # ============================================================
    # 3') MQE posterior AFTER HBM (from q_mqe_any)
    #      Uses: q_mqe_any [N,M], seg_mqe_mask [T_seg,M], MQE_NAMES
    # ============================================================

    # Dataset-level P(MQE_m = 1) under the full hierarchical model:
    P_M1_after = q_mqe_any.mean(axis=0)   # [M]

    # Only MQEs that actually appear in at least one segment
    relevant_mask = seg_mqe_mask.any(axis=0)   # [M]
    idx_rel = np.where(relevant_mask)[0]

    mqe_names_rel = [MQE_NAMES[j] for j in idx_rel]
    P_M1_rel_after = P_M1_after[idx_rel]

    plt.figure(figsize=(8, 4))
    x = np.arange(len(idx_rel))
    plt.bar(x, P_M1_rel_after)
    plt.xticks(x, mqe_names_rel, rotation=45, ha='right')
    plt.ylim(0.0, 1.0)
    plt.ylabel("P(MQE_m = 1) (dataset average)")
    plt.xlabel("Movement Quality Elements (relevant only)")
    plt.title("MQE posterior after HBM (full T→S→MQE)")
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


    print('done')
 
    ###################################statistical tests##########################################
    # ------------------------------------------------------------
    # Helper: print summary stats
    # ------------------------------------------------------------
    def summarize_entropy(name, H):
        print(f"\n[{name}]")
        print(f"  N    = {H.size}")
        print(f"  mean = {H.mean():.6f}")
        print(f"  std  = {H.std(ddof=1):.6f}")
        print(f"  min  = {H.min():.6f}")
        print(f"  max  = {H.max():.6f}")
    # ------------------------------------------------------------
    # 1) Compare live vs video task entropy (independent t-test)
    # ------------------------------------------------------------
    # H_task_live : entropy from live_rating_cleaned (one clinician)
    # H_task      : entropy from video-based task ratings (DS on R_task)

    summarize_entropy("Live pre-intervention (task-only)", H_task_live)
    summarize_entropy("Video ratings (task-only DS)",      H_task)
   
    # Welch's t-test (does not assume equal variance)
    tt_ind = stats.ttest_ind(H_task_live, H_task, equal_var=False)

    # Cohen's d for independent samples
    mean_diff = H_task_live.mean() - H_task.mean()
    # pooled SD (unbiased, for Cohen's d)
    sd_pooled = np.sqrt(
        ((H_task_live.size - 1) * H_task_live.var(ddof=1) +
        (H_task.size      - 1) * H_task.var(ddof=1)) /
        (H_task_live.size + H_task.size - 2)
    )
    cohen_d = mean_diff / sd_pooled if sd_pooled > 0 else np.nan

    print("\n=== Independent t-test: Live vs Video (task entropy) ===")
    print(f"t-statistic = {tt_ind.statistic:.4f}")
    print(f"p-value     = {tt_ind.pvalue:.4e}")
    print(f"Cohen's d   = {cohen_d:.4f}  (effect size)")

    # ------------------------------------------------------------
    # 2) OPTIONAL: paired t-test if you align trials 1:1
    # ------------------------------------------------------------
    # If (and only if) you know that the first N_common entries correspond
    # to the same tasks in both arrays (e.g., you sort by trial ID and align),
    # you can use a paired test. Otherwise, skip this block.

    N_common = min(H_task_live.size, H_task.size)
    H_live_aligned  = H_task_live[:N_common]
    H_video_aligned = H_task[:N_common]

    tt_paired = stats.ttest_rel(H_live_aligned, H_video_aligned)

    print("\n=== Paired t-test (first N_common tasks, live vs video) ===")
    print(f"N_common    = {N_common}")
    print(f"t-statistic = {tt_paired.statistic:.4f}")
    print(f"p-value     = {tt_paired.pvalue:.4e}")

    # ------------------------------------------------------------
    # 3) OPTIONAL: compare live vs full hierarchical model (post-intervention)
    # ------------------------------------------------------------
    # If you want to show the impact of the *whole* UMERA/HBM pipeline,
    # you can also compare H_task_live vs H_task_full:

    if 'H_task_full' in globals():
        summarize_entropy("Full T→S→MQE (post-intervention)", H_task_full)

        tt_ind_full = stats.ttest_ind(H_task_live, H_task_full, equal_var=False)

        mean_diff_full = H_task_live.mean() - H_task_full.mean()
        sd_pooled_full = np.sqrt(
            ((H_task_live.size - 1) * H_task_live.var(ddof=1) +
            (H_task_full.size - 1) * H_task_full.var(ddof=1)) /
            (H_task_live.size + H_task_full.size - 2)
        )
        cohen_d_full = mean_diff_full / sd_pooled_full if sd_pooled_full > 0 else np.nan

        print("\n=== Independent t-test: Live vs Full UMERA/HBM (task entropy) ===")
        print(f"t-statistic = {tt_ind_full.statistic:.4f}")
        print(f"p-value     = {tt_ind_full.pvalue:.4e}")
        print(f"Cohen's d   = {cohen_d_full:.4f}")
    groups = [H_task_live, H_task]
    labels = ["Live (pre-intervention)", "Video/UMERA"]

    # recompute Welch t-test
    tt_ind = stats.ttest_ind(H_task_live, H_task, equal_var=False)
    t_val = tt_ind.statistic
    p_val = tt_ind.pvalue

    plt.figure(figsize=(7, 5))

    # Violin plot
    parts = plt.violinplot(groups, showmeans=True, showmedians=False, showextrema=False)

    # Customize a bit
    for pc in parts['bodies']:
        pc.set_alpha(0.4)
    if 'cmeans' in parts:
        parts['cmeans'].set_color('k')
        parts['cmeans'].set_linewidth(2)

    # Overlay boxplots
    plt.boxplot(groups, positions=[1, 2], widths=0.15, showfliers=False)

    plt.xticks([1, 2], labels, rotation=15)
    plt.ylabel("Task entropy H(T)")
    plt.title("Task entropy: live vs video/UMERA")

    # Annotate t-test result
    plt.text(
        1.5,
        max(H_task_live.max(), H_task.max()) * 1.02,
        f"Welch t = {t_val:.2f}, p < 1e-16",
        ha="center",
        va="bottom"
    )

    plt.ylim(0, max(H_task_live.max(), H_task.max()) * 1.1)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    def mean_ci_95(x):
        x = np.asarray(x)
        n = x.size
        m = x.mean()
        se = x.std(ddof=1) / np.sqrt(n)
        ci = 1.96 * se
        return m, ci

    m_live, ci_live   = mean_ci_95(H_task_live)
    m_video, ci_video = mean_ci_95(H_task)

    means = [m_live, m_video]
    cis   = [ci_live, ci_video]
    labels = ["Live (pre-intervention)", "Video/UMERA"]

    x = np.arange(2)

    plt.figure(figsize=(6, 5))
    plt.bar(x, means, yerr=cis, capsize=6, alpha=0.8)
    plt.xticks(x, labels, rotation=15)
    plt.ylabel("Mean task entropy H(T)")
    plt.title("Mean uncertainty with 95% CI")

    # add numeric text on top
    for i, (m, ci) in enumerate(zip(means, cis)):
        plt.text(i, m + ci + 0.02, f"{m:.3f}±{ci:.3f}", ha="center", va="bottom")

    plt.ylim(0, max(m_live + ci_live, m_video + ci_video) + 0.1)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()
