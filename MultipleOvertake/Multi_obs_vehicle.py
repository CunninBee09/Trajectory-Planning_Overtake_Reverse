import copy
import math
import numpy as np
from MultipleOvertake.Multi_ego_vehicle import calc_frenet_paths, calc_global_paths
from utilis import *

def obstacle_planning(csp, obs_s0, obs_speed, obs_acc, obs_d, obs_d_d, obs_d_dd, Target_obs_speed):
    obs_dict= calc_frenet_paths(obs_speed, obs_acc, obs_d, obs_d_d, obs_d_dd, obs_s0, Target_obs_speed)
    obs_dict= calc_global_paths(obs_dict, csp)
    obs_dict= check_obs_paths(obs_dict)
    
    # find minimum cost path
    min_cost = float("inf")
    best_obs_path = None
    for di,obslist in obs_dict.items():
        for ob in obslist:
            if min_cost >= ob.cf:
                min_cost = ob.cf
                best_obs_path = ob

    return best_obs_path

def check_obs_paths(obs_dict):
    if obs_dict is None:
        return None

    filtered_obs_dict = {}  # To store valid paths categorized by road width

    for di, obslist in obs_dict.items():  # Iterate over the dictionary
        ok_ind = []  # Indices of valid paths for the current road width

        for i, ob in enumerate(obslist):
            if any([v > MAX_OBS_SPEED for v in ob.s_d]):  # Max speed check
                continue
            elif any([abs(a) > MAX_OBS_ACCEL for a in ob.s_dd]):  # Max accel check
                continue
            elif any([abs(c) > MAX_CURVATURE for c in ob.c]):  # Max curvature check
                continue
            ok_ind.append(i)
            
        if ok_ind:
            # Store only the valid paths for this road width
            filtered_obs_dict[di] = [obslist[i] for i in ok_ind]

    if not filtered_obs_dict:
        print("All candidate paths failed constraints!")
        return None

    return filtered_obs_dict
