import torch
import numpy as np
from tqdm import tqdm

def nh_sample_actions(n_actions: int, rng, action_params: dict) -> torch.tensor:
    relative_x_offset = rng.normal(loc=action_params["relative_x_offset_mean"], scale=action_params["relative_x_offset_std"], size=n_actions)[:,None]
    relative_y_offset = rng.normal(loc=action_params["relative_y_offset_mean"], scale=action_params["relative_y_offset_std"], size=n_actions)[:,None]
    theta_offset = rng.uniform(low=np.deg2rad(action_params["theta_offset_low"]), high=np.deg2rad(action_params["theta_offset_high"]), size=n_actions)[:,None]

    actions = np.concatenate([relative_x_offset, relative_y_offset, theta_offset], axis=-1)
    return torch.from_numpy(actions)

def nh_apply_action(pose_1: torch.tensor, action: torch.tensor) -> torch.tensor:
    ## ToDo: Doc-string, comment ##
    pose_2 = torch.zeros_like(pose_1)

    x_1 = pose_1[0]
    y_1 = pose_1[1]
    theta_1 = pose_1[2]

    relative_x_offset = action[0]
    relative_y_offset = action[1]
    delta_theta = action [2]

    offset_distance = torch.sqrt(torch.pow(relative_x_offset, 2) + torch.pow(relative_y_offset, 2))
    if offset_distance == 0:
        beta = 0
    else:
        beta = torch.asin(relative_y_offset / offset_distance)  # changed from acos to asin 18.11.2022
    alpha = theta_1 + beta  # changed from - beta to + beta 03.12.2022

    delta_x = torch.cos(alpha) * offset_distance
    delta_y = torch.sin(alpha) * offset_distance

    x_2 = x_1 + delta_x
    y_2 = y_1 + delta_y
    theta_2 = theta_1 + delta_theta

    if theta_2 > (2*torch.pi):
        theta_2 = theta_2 - (2*torch.pi)
    elif theta_2 < 0:
        theta_2 = (2*torch.pi) + theta_2
    else:
        theta_2 = theta_2

    pose_2[0] = x_2
    pose_2[1] = y_2
    pose_2[2] = theta_2

    return pose_2

def nh_normalize(val, min, max):
    return (val-min)/(max-min)

def nh_denorm(val, min, max):
    return (val*(max-min))+min

def nh_find_closest_pose(new_pose: torch.tensor, poses: torch.tensor, sample_ids: torch.tensor, 
    h_embedding_ids: torch.tensor, beamrs: torch.tensor, rng, pose_weights: dict) -> tuple:

    ## ToDo: Doc-string, comment ##
    all_positions = poses[:, :2]
    new_position = new_pose[:2]
    distances = torch.sqrt(torch.pow(all_positions-new_position, 2).sum(dim=-1))

    all_orientations = poses[:, 2]
    new_orientation = new_pose[2]
    angle_diffs = torch.abs(((all_orientations-new_orientation+(3*torch.pi))%(2*torch.pi)) - torch.pi) # interval: [-pi, pi)

    normalized_distances =  nh_normalize(distances, 0, np.sqrt(20)) # sqrt(20) is the maximal distance (diagonal) of two points in the subroom
    normalized_angle_diffs = nh_normalize(angle_diffs, 0, torch.pi) # define norm_func

    score = (pose_weights["distance_weight"]*normalized_distances) + (pose_weights["angle_weight"]*normalized_angle_diffs)

    random_winner_choice = rng.choice(3, 1, p=[0.5, 0.3, 0.2])  # make candidate number and probabilities a parameter
    winner_idx = torch.argsort(score)[random_winner_choice].item()
    
    #winner_idx = torch.argmin(score)   # sort in ascending order and draw from gamma-distribution in order for states not to get stuck

    closest_pose = poses[winner_idx]
    sample_id = sample_ids[winner_idx]
    h_embedding_id = h_embedding_ids[winner_idx]
    beamr = beamrs[winner_idx]

    return closest_pose, sample_id, h_embedding_id, beamr

def nh_calculate_action_between_poses(pose_1: torch.tensor, pose_2: torch.tensor) -> torch.tensor:
    ## ToDo: Doc-string, comment ##
    x1 = pose_1[0]
    y1 = pose_1[1]
    t1 = pose_1[2]

    x2 = pose_2[0]
    y2 = pose_2[1]
    t2 = pose_2[2]

    rho = ((x2-x1)**2 + (y2-y1)**2)**0.5
    rho_angle = torch.atan2((y2-y1), (x2-x1)) # error use atan2
    alpha = ((t1-rho_angle+(3*torch.pi))%(2*torch.pi)) - torch.pi
    beta = ((t2-t1-alpha+(3*torch.pi))%(2*torch.pi)) - torch.pi

    x_r = rho * torch.cos(alpha)
    y_r = rho * torch.sin(alpha)
    delta_t = ((t2-t1+(3*torch.pi))%(2*torch.pi)) - torch.pi

    action = torch.tensor([x_r, y_r, delta_t])
    return action

def nh_sample_sequence(pose: torch.tensor, sample_id: int, h_embedding_id: int, poses: torch.tensor, action: torch.tensor, 
    sample_ids: torch.tensor, h_embedding_ids: torch.tensor, seq_len: int, beamr: torch.tensor, beamrs: torch.tensor, rng, pose_weights: dict) -> tuple:

    pose_list = [torch.unsqueeze(pose, dim=0)]
    sample_id_list = [sample_id]
    h_embedding_id_list = [h_embedding_id]
    action_list = []
    beamr_list = [torch.unsqueeze(beamr, dim=0)]

    current_pose = pose
    for i in range(seq_len-1):
        # maybe slightly alter action in order for states not to get stuck
        new_pose = nh_apply_action(current_pose, action)   # new calculated pose
        closest_pose, closest_sample_id, closest_h_embedding_id, closest_beamr = nh_find_closest_pose(
            new_pose, poses, sample_ids, h_embedding_ids, beamrs, rng, pose_weights
        )  # closest matching pose from existing poses
        adjusted_action = nh_calculate_action_between_poses(current_pose, closest_pose)    # get action from current pose to closest matching pose

        pose_list.append(torch.unsqueeze(closest_pose, dim=0))
        sample_id_list.append(closest_sample_id)
        h_embedding_id_list.append(closest_h_embedding_id)
        action_list.append(torch.unsqueeze(adjusted_action, dim=0))
        beamr_list.append(torch.unsqueeze(closest_beamr, dim=0))

        current_pose = closest_pose
    
    return (torch.cat(pose_list, axis=0), torch.tensor(sample_id_list), torch.tensor(h_embedding_id_list), 
            torch.cat(action_list, axis=0), torch.cat(beamr_list, axis=0))

def nh_get_sequences(poses: torch.tensor, beamrs: torch.tensor, sample_ids: torch.tensor, 
    h_embedding_ids: torch.tensor, n_actions: int, seq_len: int, rng, action_params: dict, pose_weights: dict) -> tuple: 

    pose_sequence_list = []
    sample_id_sequence_list = []
    h_embedding_id_sequence_list = []
    action_sequence_list = []
    beamr_sequence_list = []

    for pose, sample_id, beamr, h_embedding_id in tqdm(zip(poses, sample_ids, beamrs, h_embedding_ids)):
        actions = nh_sample_actions(n_actions, rng, action_params)  # get list of actions from current pose
        for action in actions:
            pose_sequence, sample_id_sequence, h_embedding_id_sequence, action_sequence, beamr_sequence = nh_sample_sequence(
                pose, sample_id, h_embedding_id, poses, action, sample_ids, h_embedding_ids, seq_len, beamr, beamrs, rng, pose_weights
            )    # sample sequence from current pose and given action vector

            pose_sequence_list.append(torch.unsqueeze(pose_sequence, dim=0))
            sample_id_sequence_list.append(torch.unsqueeze(sample_id_sequence, dim=0))
            h_embedding_id_sequence_list.append(torch.unsqueeze(h_embedding_id_sequence, dim=0))
            action_sequence_list.append(torch.unsqueeze(action_sequence, dim=0))
            beamr_sequence_list.append(torch.unsqueeze(beamr_sequence, dim=0))

    return (torch.cat(pose_sequence_list, axis=0), torch.cat(sample_id_sequence_list, axis=0), torch.cat(h_embedding_id_sequence_list, axis=0), 
            torch.cat(action_sequence_list, axis=0), torch.cat(beamr_sequence_list, axis=0))

def nh_normalize_0_1(poses: torch.tensor, actions: torch.tensor, beamrs: torch.tensor, 
    action_trans_min: float, action_trans_max: float, action_angle_min: float, action_angle_max: float, 
    distance_min: float, distance_max: float, angle_min: float, angle_max: float, pos_min: float, pos_max: float) -> tuple:

    poses_norm = torch.empty_like(poses)
    actions_norm = torch.empty_like(actions)
    beamrs_norm = torch.empty_like(beamrs)

    beamrs_norm = nh_normalize(beamrs, distance_min, distance_max)

    for i in range(2):
        poses_norm[:, :, i] = nh_normalize(poses[:, :, i], pos_min, pos_max)
        actions_norm[:, :, i] = nh_normalize(actions[:, :, i], action_trans_min, action_trans_max)

    poses_norm[:, :, 2] = nh_normalize(poses[:, :, 2], angle_min, angle_max)
    actions_norm[:, :, 2] = nh_normalize(actions[:, :, 2], action_angle_min, action_angle_max)

    return (poses_norm, actions_norm, beamrs_norm) 