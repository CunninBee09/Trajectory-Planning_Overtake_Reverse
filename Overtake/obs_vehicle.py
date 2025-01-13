import copy
import math
import numpy as np
from ego_vehicle import calc_frenet_paths, calc_global_paths
from utilis import *

def obstacle_planning(csp, obs_s0, obs_speed, obs_acc, obs_d, obs_d_d, obs_d_dd, Target_obs_speed):
    obslist= calc_frenet_paths(obs_speed, obs_acc, obs_d, obs_d_d, obs_d_dd, obs_s0, Target_obs_speed)
    obslist= calc_global_paths(obslist, csp)
    obslist= check_obs_paths(obslist)
    
    # find minimum cost path
    min_cost = float("inf")
    best_obs_path = None
    for ob in obslist:
        if min_cost >= ob.cf:
            min_cost = ob.cf
            best_obs_path = ob

    return best_obs_path

def check_obs_paths(obslist):
    ok_ind = []
    for i, _ in enumerate(obslist):
        if any([v > MAX_OBS_SPEED for v in obslist[i].s_d]):  # Max speed check
            continue
        elif any([abs(a) > MAX_OBS_ACCEL for a in
                  obslist[i].s_dd]):  # Max accel check
            continue
        elif any([abs(c) > MAX_CURVATURE for c in
                  obslist[i].c]):  # Max curvature check
            continue

        ok_ind.append(i)

    return [obslist[i] for i in ok_ind]