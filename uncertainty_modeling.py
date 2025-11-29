"""
umera_uncertainty_model.py

Hierarchical Bayesian label model for stroke rehab assessments.

We assume:
- N_instances = number of patient–task executions (here ~1800)
- Each instance corresponds to one of 15 ARAT task types (task_type_id in {0..14})
- Each instance has 4 segments (e.g., init, grasp, transport, release)
- There are C = 2 clinicians
- There are M MQE types (e.g., 10 movement quality elements)

Data you provide:
- R_task: [N_instances, C] integer array in {0,1,2,3} or -1 for missing
- R_seg:  [N_instances, T_seg=4, C] integer array in {0,1,2,3} or -1 for missing
- R_mqe:  [N_instances, T_seg=4, M, C] integer array in {0,1} or -1 for missing
- task_type_id: [N_instances] integer array in {0..14}

We build:
  1) Task-only label model
  2) Task + segment model
  3) Task + segment + MQE model

and provide utilities to compare uncertainty (entropy) across them.
"""
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
    Loads the task_final and segment_final CSV files containing ratings.
    
    Expected files in csv_dir:
      - task_final.csv with columns: PatientTaskHandMappingId, Completed, Initialized, Time, Impaired, Rating, TherapistId, CreatedAt, ModifiedAt, Finger
      - segment_final.csv with columns: PatientTaskHandMappingId, SegmentId, Completed, Initialized, Time, Impaired, Rating, TherapistId, CreatedAt, ModifiedAt, Finger
      
    Returns:
      Two dictionaries:
         task_ratings: mapping PatientTaskHandMappingId to a dictionary with keys 't1' and (optionally) 't2'
                       for task ratings.
         segment_ratings: mapping PatientTaskHandMappingId to a dictionary with keys 't1' and (optionally) 't2',
                          where each value is itself a dictionary mapping SegmentId to its rating.
    """   
    task_file = os.path.join(csv_dir, "task_final_updated.csv")
    segment_file = os.path.join(csv_dir, "segment_final_updated.csv")
    
    task_df = pd.read_csv(task_file)
    segment_df = pd.read_csv(segment_file)
    
    # Process task ratings: store first rating as 't1' and second (if available) as 't2'.
    task_ratings = {}
    for _, row in task_df.iterrows():
        mapping_id = row['PatientTaskHandMappingId']
        rating = row['Rating']
        if pd.notna(rating):
            if mapping_id not in task_ratings:
                task_ratings[mapping_id] = {'t1': rating}
            elif 't2' not in task_ratings[mapping_id]:
                task_ratings[mapping_id]['t2'] = rating
            # Ignore any additional ratings.
    
    # Process segment ratings:
    # For each mapping id and therapist, build a dictionary mapping each SegmentId to its rating.
    segment_ratings = {}
    grouped = segment_df.groupby(['PatientTaskHandMappingId', 'TherapistId'])
    for (mapping_id, therapist_id), group in grouped:
        seg_rating_dict = {}
        for _, row in group.iterrows():
            # Assuming segment_final.csv has a 'SegmentId' column.
            seg_id = row['SegmentId']
            rating = row['Rating']
            if pd.notna(rating):
                seg_rating_dict[seg_id] = rating
        if not seg_rating_dict:
            continue
        # For each mapping_id, store the first therapist's segment ratings as 't1'
        # and if a second therapist is available, store their ratings as 't2'.
        if mapping_id not in segment_ratings:
            segment_ratings[mapping_id] = {'t1': seg_rating_dict}
        elif 't1' in segment_ratings[mapping_id] and 't2' not in segment_ratings[mapping_id]:
            segment_ratings[mapping_id]['t2'] = seg_rating_dict
        # If already both t1 and t2 exist, ignore extra groups.
    
    return task_ratings, segment_ratings
import numpy as np
import pymc as pm
import arviz as az
import pytensor
import pytensor.tensor as pt
import os
import pandas as pd
import pytensor

# ---------------------------------------------------------------------
# ENV WORKAROUND: avoid MinGW C++ compile issues on Windows
# ---------------------------------------------------------------------
pytensor.config.cxx = ""              # keep this so it doesn't try to use bad MinGW
pytensor.config.mode = "FAST_RUN"     # or comment this out entirely


# ---------------------------------------------------------------------
# Helpers: posterior probs / entropy / margin
# ---------------------------------------------------------------------

def posterior_probs_T(idata, var_name="T", K=4):
    """
    Compute posterior probabilities p(T_i = y | data) from MCMC samples.

    idata.posterior[var_name] shape: [chain, draw, N_instances]
    Returns: probs [N_instances, K]
    """
    T_samples = idata.posterior[var_name].values  # [chain, draw, N]
    T_samples = np.reshape(T_samples, (-1, T_samples.shape[-1]))  # [samples, N]
    n_samples, N = T_samples.shape
    probs = np.zeros((N, K), dtype=float)
    for y in range(K):
        probs[:, y] = (T_samples == y).mean(axis=0)
    return probs


def entropy(p, eps=1e-12):
    """
    Shannon entropy in nats for distributions p along last axis.
    p: [N, K] or [..., K]
    Returns: H: [N] or [...]
    """
    p_clipped = np.clip(p, eps, 1.0 - eps)
    return -(p_clipped * np.log(p_clipped)).sum(axis=-1)


def margin(p):
    """
    Confidence margin per sample:
    margin = p(max class) - p(second-best class)
    p: [N, K]
    Returns: [N]
    """
    sorted_p = np.sort(p, axis=1)
    best = sorted_p[:, -1]
    second_best = sorted_p[:, -2]
    return best - second_best


# ---------------------------------------------------------------------
# Model 1: Task-only label model
# ---------------------------------------------------------------------

def build_model_task_only(R_task, task_type_id=None, K=4, n_task_types=15):
    """
    Task-only Bayesian label model:

    Latent T[i] in {0..K-1} for each instance i.
    Two clinicians, each with their own KxK confusion matrix for task-level ratings.

    Optionally model task-type-specific priors:
      pi_task_type[j, :] for j in {0..n_task_types-1},
      so T[i] ~ Categorical(pi_task_type[task_type_id[i]])

    Parameters
    ----------
    R_task : [N_instances, C]
        Task ratings from clinicians (0..K-1 or -1 for missing).
    task_type_id : [N_instances] or None
        Integer task type for each instance in {0..n_task_types-1}.
        If None, use a global Dirichlet prior for T.
    K : int
        Number of ARAT categories (0..3).
    n_task_types : int
        Number of distinct ARAT task types (here 15).

    Returns
    -------
    model : pm.Model
    """
    R_task = np.asarray(R_task, dtype=int)
    N_instances, C = R_task.shape

    if task_type_id is not None:
        task_type_id = np.asarray(task_type_id, dtype=int)
        assert task_type_id.shape[0] == N_instances

    with pm.Model() as model:
        # -----------------------------------------
        # Prior over true task scores T[i]
        # -----------------------------------------
        if task_type_id is None:
            # Single global prior over T
            pi_task = pm.Dirichlet("pi_task", a=np.ones(K))
            T = pm.Categorical("T", p=pi_task, shape=N_instances)
        else:
            # Task-type-specific priors: pi_task_type[task_type, y]
            pi_task_type = pm.Dirichlet(
                "pi_task_type", a=np.ones((n_task_types, K)), shape=(n_task_types, K)
            )
            # For each instance i, use pi_task_type[task_type_id[i]]
            pi_task_i = pi_task_type[task_type_id]  # shape [N, K]
            T = pm.Categorical("T", p=pi_task_i, shape=N_instances)

        # -----------------------------------------
        # Clinician task confusion matrices
        # -----------------------------------------
        # Theta_task[c, true, observed]
        base_alpha = np.eye(K) * 4.0 + 1.0  # mildly diagonal
        a_task_prior = np.broadcast_to(base_alpha, (C, K, K))
        Theta_task = pm.Dirichlet(
            "Theta_task", a=a_task_prior, shape=(C, K, K)
        )

        # -----------------------------------------
        # Likelihood: observed task ratings
        # -----------------------------------------
        for i in range(N_instances):
            for c in range(C):
                r = R_task[i, c]
                if r >= 0:
                    pm.Categorical(
                        f"R_task_{i}_{c}",
                        p=Theta_task[c, T[i], :],
                        observed=r,
                    )

    return model


# ---------------------------------------------------------------------
# Model 2: Task + segments
# ---------------------------------------------------------------------

def build_model_task_segments(R_task, R_seg, task_type_id=None,
                              K=4, T_seg=4, n_task_types=15):
    """
    Hierarchical label model with:
      - Latent segment scores S[i, t] in {0..K-1},
      - Latent task scores T[i] depending on segment score frequencies,
      - Clinician confusion matrices for task and segment ratings.

    Here:
      N_instances = R_task.shape[0],
      T_seg = 4 segments per instance (e.g., init, grasp, transport, release),
      C = number of clinicians (here 2).

    Parameters
    ----------
    R_task : [N_instances, C]
        Task ratings (0..K-1 or -1).
    R_seg : [N_instances, T_seg, C]
        Segment ratings (0..K-1 or -1).
    task_type_id : [N_instances] or None
        Task type ID per instance in {0..n_task_types-1}.
        We use it to allow segment distributions to vary by task type if desired.
    K : int
        Number of ARAT categories.
    T_seg : int
        Number of segments per instance (here 4).
    n_task_types : int
        Number of ARAT task types (15).

    Returns
    -------
    model : pm.Model
    """
    R_task = np.asarray(R_task, dtype=int)
    R_seg = np.asarray(R_seg, dtype=int)

    N_instances, C = R_task.shape
    N2, T_seg2, C2 = R_seg.shape
    assert N_instances == N2 and C == C2
    assert T_seg == T_seg2

    if task_type_id is not None:
        task_type_id = np.asarray(task_type_id, dtype=int)
        assert task_type_id.shape[0] == N_instances

    with pm.Model() as model:
        # -----------------------------------------
        # 1. Segment priors S[i, t]
        # -----------------------------------------
        # Option A: same segment distribution for all task types.
        # Option B: segment distribution depends on task type & segment index.
        # For simplicity, we do B:
        #   pi_seg[task_type, segment_index, score]
        pi_seg = pm.Dirichlet(
            "pi_seg",
            a=np.ones((n_task_types, T_seg, K)),
            shape=(n_task_types, T_seg, K),
        )

        if task_type_id is None:
            # fall back to a single generic pi_seg, ignoring task type
            # but use only first entry [0] for modeling
            pi_seg_i = pi_seg[0]  # [T_seg, K]
            pi_seg_i = pt.repeat(pi_seg_i[None, :, :], N_instances, axis=0)
        else:
            # For each instance i, use pi_seg[task_type_i, t, :]
            pi_seg_i = pi_seg[task_type_id, :, :]  # [N_instances, T_seg, K]

        S = pm.Categorical(
            "S",
            p=pi_seg_i,
            shape=(N_instances, T_seg)
        )

        # -----------------------------------------
        # 2. Task scores T[i] given segments S[i, :]
        # -----------------------------------------
        # One-hot encode S: shape (N_instances, T_seg, K)
        S_one_hot = pt.eye(K)[S]
        seg_counts = S_one_hot.sum(axis=1)       # (N_instances, K)
        seg_freqs = seg_counts / T_seg           # (N_instances, K)

        # Linear model: logits_task[i, y] = W_task[y, :] @ seg_freqs[i, :] + b_task[y]
        W_task = pm.Normal("W_task", mu=0.0, sigma=1.0, shape=(K, K))
        b_task = pm.Normal("b_task", mu=0.0, sigma=1.0, shape=(K,))
        logits_task = pt.dot(seg_freqs, W_task.T) + b_task
        pi_task = pt.nnet.softmax(logits_task)

        T = pm.Categorical("T", p=pi_task, shape=N_instances)

        # -----------------------------------------
        # 3. Confusion matrices for task & segment ratings
        # -----------------------------------------
        base_alpha = np.eye(K) * 4.0 + 1.0

        a_task_prior = np.broadcast_to(base_alpha, (C, K, K))
        Theta_task = pm.Dirichlet(
            "Theta_task", a=a_task_prior, shape=(C, K, K)
        )

        a_seg_prior = np.broadcast_to(base_alpha, (C, K, K))
        Theta_seg = pm.Dirichlet(
            "Theta_seg", a=a_seg_prior, shape=(C, K, K)
        )

        # -----------------------------------------
        # 4. Likelihood: task ratings
        # -----------------------------------------
        for i in range(N_instances):
            for c in range(C):
                r = R_task[i, c]
                if r >= 0:
                    pm.Categorical(
                        f"R_task_{i}_{c}",
                        p=Theta_task[c, T[i], :],
                        observed=r,
                    )

        # -----------------------------------------
        # 5. Likelihood: segment ratings
        # -----------------------------------------
        for i in range(N_instances):
            for t in range(T_seg):
                for c in range(C):
                    r = R_seg[i, t, c]
                    if r >= 0:
                        pm.Categorical(
                            f"R_seg_{i}_{t}_{c}",
                            p=Theta_seg[c, S[i, t], :],
                            observed=r,
                        )

    return model


# ---------------------------------------------------------------------
# Model 3: Task + segments + MQEs
# ---------------------------------------------------------------------

def build_model_task_segments_mqe(R_task, R_seg, R_mqe, task_type_id=None,
                                  K=4, T_seg=4, M_mqe=10, n_task_types=15):
    """
    Full hierarchical label model:

      - Latent segment scores S[i, t]
      - Latent task scores T[i] depending on S[i,:]
      - Latent MQEs Z[i, t, m] depending on S[i, t]
      - Rater-specific confusion matrices for task and segment scores
      - Rater-specific sensitivities/specificities for MQEs

    Shapes:
      R_task : [N_instances, C]
      R_seg  : [N_instances, T_seg, C]
      R_mqe  : [N_instances, T_seg, M_mqe, C]
      task_type_id : [N_instances] in {0..n_task_types-1}

    Returns
    -------
    model : pm.Model
    """
    R_task = np.asarray(R_task, dtype=int)
    R_seg = np.asarray(R_seg, dtype=int)
    R_mqe = np.asarray(R_mqe, dtype=int)

    N_instances, C = R_task.shape
    N2, T_seg2, C2 = R_seg.shape
    N3, T_seg3, M, C3 = R_mqe.shape

    assert N_instances == N2 == N3
    assert C == C2 == C3
    assert T_seg == T_seg2 == T_seg3
    assert M_mqe == M

    if task_type_id is not None:
        task_type_id = np.asarray(task_type_id, dtype=int)
        assert task_type_id.shape[0] == N_instances

    with pm.Model() as model:
        # -----------------------------------------
        # 1. Segment priors S[i, t]
        # -----------------------------------------
        pi_seg = pm.Dirichlet(
            "pi_seg",
            a=np.ones((n_task_types, T_seg, K)),
            shape=(n_task_types, T_seg, K),
        )

        if task_type_id is None:
            pi_seg_i = pi_seg[0]
            pi_seg_i = pt.repeat(pi_seg_i[None, :, :], N_instances, axis=0)
        else:
            pi_seg_i = pi_seg[task_type_id, :, :]  # [N_instances, T_seg, K]

        S = pm.Categorical("S", p=pi_seg_i, shape=(N_instances, T_seg))

        # -----------------------------------------
        # 2. Task scores T[i] given S[i,:]
        # -----------------------------------------
        S_one_hot = pt.eye(K)[S]          # (N_instances, T_seg, K)
        seg_counts = S_one_hot.sum(axis=1)  # (N_instances, K)
        seg_freqs = seg_counts / T_seg      # (N_instances, K)

        W_task = pm.Normal("W_task", mu=0.0, sigma=1.0, shape=(K, K))
        b_task = pm.Normal("b_task", mu=0.0, sigma=1.0, shape=(K,))
        logits_task = pt.dot(seg_freqs, W_task.T) + b_task
        pi_task = pt.nnet.softmax(logits_task)
        T = pm.Categorical("T", p=pi_task, shape=N_instances)

        # -----------------------------------------
        # 3. MQEs Z[i,t,m] given S[i,t]
        # -----------------------------------------
        # theta_mqe[score, m] = P(Z=1 | segment_score=score, MQE type m)
        theta_mqe = pm.Beta(
            "theta_mqe",
            alpha=1.0,
            beta=1.0,
            shape=(K, M_mqe),
        )

        # Z[i, t, m] ~ Bernoulli(theta_mqe[S[i,t], m])
        Z = pm.Bernoulli(
            "Z",
            p=theta_mqe[S],
            shape=(N_instances, T_seg, M_mqe),
        )

        # -----------------------------------------
        # 4. Rater-specific parameters
        # -----------------------------------------
        base_alpha = np.eye(K) * 4.0 + 1.0

        # Task-level confusion
        a_task_prior = np.broadcast_to(base_alpha, (C, K, K))
        Theta_task = pm.Dirichlet(
            "Theta_task", a=a_task_prior, shape=(C, K, K)
        )

        # Segment-level confusion
        a_seg_prior = np.broadcast_to(base_alpha, (C, K, K))
        Theta_seg = pm.Dirichlet(
            "Theta_seg", a=a_seg_prior, shape=(C, K, K)
        )

        # MQE sensitivity/specificity per (clinician, MQE type)
        sens = pm.Beta(
            "sens",
            alpha=2.0,
            beta=2.0,
            shape=(C, M_mqe),
        )
        spec = pm.Beta(
            "spec",
            alpha=2.0,
            beta=2.0,
            shape=(C, M_mqe),
        )

        # -----------------------------------------
        # 5. Likelihood: task ratings
        # -----------------------------------------
        for i in range(N_instances):
            for c in range(C):
                r = R_task[i, c]
                if r >= 0:
                    pm.Categorical(
                        f"R_task_{i}_{c}",
                        p=Theta_task[c, T[i], :],
                        observed=r,
                    )

        # -----------------------------------------
        # 6. Likelihood: segment ratings
        # -----------------------------------------
        for i in range(N_instances):
            for t in range(T_seg):
                for c in range(C):
                    r = R_seg[i, t, c]
                    if r >= 0:
                        pm.Categorical(
                            f"R_seg_{i}_{t}_{c}",
                            p=Theta_seg[c, S[i, t], :],
                            observed=r,
                        )

        # -----------------------------------------
        # 7. Likelihood: MQE annotations
        # -----------------------------------------
        # R_mqe[i, t, m, c] in {0, 1, -1}, -1 = missing
        for i in range(N_instances):
            for t in range(T_seg):
                for m_idx in range(M_mqe):
                    for c in range(C):
                        r = R_mqe[i, t, m_idx, c]
                        if r >= 0:
                            # P(R=1 | Z, sens, spec) = Z*sens + (1-Z)*(1-spec)
                            p_obs = (
                                Z[i, t, m_idx] * sens[c, m_idx]
                                + (1 - Z[i, t, m_idx]) * (1 - spec[c, m_idx])
                            )
                            pm.Bernoulli(
                                f"R_mqe_{i}_{t}_{m_idx}_{c}",
                                p=p_obs,
                                observed=r,
                            )

    return model


# ---------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------

import pymc as pm

def run_inference(model, draws=100, tune=100, chains=1,
                  target_accept=0.9, progressbar=True):
    with model:
        print(f"[run_inference] Starting: draws={draws}, tune={tune}, chains={chains}")
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            cores=1,              # no multiprocessing on Windows
            progressbar=progressbar,
            return_inferencedata=True,
        )
        print("[run_inference] Finished sampling")
    return idata


def compute_task_entropy_from_idata(idata, K=4, var_name="T"):
    """
    Convenience helper: get p(T), entropy, and margin from inference.
    """
    probs_T = posterior_probs_T(idata, var_name=var_name, K=K)
    H_task = entropy(probs_T)
    margins = margin(probs_T)
    return probs_T, H_task, margins


# ---------------------------------------------------------------------
# EXAMPLE STUB (replace with your real data)
# ---------------------------------------------------------------------
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
    task_ratings_dict, segment_ratings_dict = load_rating_info(csv_dir)
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

    # Fake MQE annotations: 0/1 per segment, type, clinician
    R_mqe = np.random.choice(
        [0, 1],
        size=(N_instances, T_seg, M_mqe, C),
        p=[0.7, 0.3],
    )
    mask_mqe = np.random.rand(N_instances, T_seg, M_mqe, C) < 0.2
    R_mqe[mask_mqe] = -1

    # -------------------- 1) Task-only --------------------
    print("Building task-only model...")
    model_task_only = build_model_task_only(
        R_task, task_type_id=task_type_id, K=K, n_task_types=n_task_types
    )
    print("Sampling task-only model...")
    idata_task_only = run_inference(
        model_task_only,
        draws=100,
        tune=100,
        chains=1,
        progressbar=True,
    )
    print('computing entropy for task only model')
    probs_T_task_only, H_task_only, margins_only = compute_task_entropy_from_idata(
        idata_task_only, K=K
    )

    # -------------------- 2) Task + segments --------------------
    print("Building task+segments model...")
    model_ts = build_model_task_segments(
        R_task, R_seg, task_type_id=task_type_id, K=K, T_seg=T_seg, n_task_types=n_task_types
    )
    print("Sampling task+segments model...")
    idata_ts = run_inference(model_ts, draws=500, tune=500, chains=2)
    probs_T_ts, H_task_ts, margins_ts = compute_task_entropy_from_idata(
        idata_ts, K=K
    )

    # -------------------- 3) Task + segments + MQEs --------------------
    print("Building full model (task+segments+MQE)...")
    model_full = build_model_task_segments_mqe(
        R_task, R_seg, R_mqe,
        task_type_id=task_type_id,
        K=K, T_seg=T_seg, M_mqe=M_mqe, n_task_types=n_task_types
    )
    print("Sampling full model...")
    idata_full = run_inference(model_full, draws=500, tune=500, chains=2)
    probs_T_full, H_task_full, margins_full = compute_task_entropy_from_idata(
        idata_full, K=K
    )

    # -------------------- 4) Uncertainty comparison --------------------
    print("\nAverage task entropy (task-only):           ", H_task_only.mean())
    print("Average task entropy (task+segments):        ", H_task_ts.mean())
    print("Average task entropy (task+seg+MQE):         ", H_task_full.mean())

    delta_H_seg = H_task_only - H_task_ts
    delta_H_mqe = H_task_ts - H_task_full

    print("\nAverage entropy reduction from segments:     ", delta_H_seg.mean())
    print("Average entropy reduction from MQEs:         ", delta_H_mqe.mean())
    import matplotlib.pyplot as plt

    plt.hist(H_task_full, bins=30)
    plt.xlabel("Task entropy H(T_i)")
    plt.ylabel("Number of tasks")
    plt.title("Distribution of task uncertainty across all ARAT instances")
    plt.show()

