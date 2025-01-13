import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import copy
import math
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from FrenetOptimalTrajectory.frenet_optimal_trajectory import \
    frenet_optimal_planning
from utilis import *

# from CubicSplinePlanner import cubic_spline_planner

class QuarticPolynomial:

    def __init__(self, xs, vxs, axs, vxe, axe, time):
        # calc coefficient of quartic polynomial

        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[3 * time ** 2, 4 * time ** 3],
                      [6 * time, 12 * time ** 2]])
        b = np.array([vxe - self.a1 - 2 * self.a2 * time,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + \
             self.a3 * t ** 3 + self.a4 * t ** 4

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
             3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t

        return xt

class QuinticPolynomial:

    def __init__(self, xs, vxs, axs, xe, vxe, axe, time):
        # calc coefficient of quintic polynomial
        
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[time ** 3, time ** 4, time ** 5],
                      [3 * time ** 2, 4 * time ** 3, 5 * time ** 4],
                      [6 * time, 12 * time ** 2, 20 * time ** 3]])
        b = np.array([xe - self.a0 - self.a1 * time - self.a2 * time ** 2,
                      vxe - self.a1 - 2 * self.a2 * time,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + \
             self.a3 * t ** 3 + self.a4 * t ** 4 + self.a5 * t ** 5

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
             3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3 + 5 * self.a5 * t ** 4

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2 + 20 * self.a5 * t ** 3

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t ** 2

        return xt

class FrenetPath:

    def __init__(self):
        self.t = []
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.d_ddd = []
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.s_ddd = []
        self.cd = 0.0
        self.cv = 0.0
        self.cf = 0.0

        self.x = []
        self.y = []
        self.yaw = []
        self.ds = []
        self.c = []
        
def calc_frenet_paths(c_speed, c_accel, c_d, c_d_d, c_d_dd, s0, V):
    frenet_paths_by_road_width = {}

    # generate path to each offset goal
    for di in np.arange(-MAX_ROAD_WIDTH,3*MAX_ROAD_WIDTH, D_ROAD_W):
        frenet_paths_by_road_width[di]= []

        # Lateral motion planning
        for Ti in np.arange(MIN_T, MAX_T, DT):
            fp = FrenetPath()

            # lat_qp = quintic_polynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)
            lat_qp = QuinticPolynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)

            fp.t = [t for t in np.arange(0.0, Ti, DT)]
            fp.d = [lat_qp.calc_point(t) for t in fp.t]
            fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp.t]
            fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp.t]
            fp.d_ddd = [lat_qp.calc_third_derivative(t) for t in fp.t]

            # Longitudinal motion planning (Velocity keeping)
            v_values = np.arange(V - (D_T_S * N_S_SAMPLE),V + (D_T_S * N_S_SAMPLE), D_T_S)
            pos_V = [float(v) for v in v_values if v>=0]
            for tv in pos_V:
                tfp = copy.deepcopy(fp)
                lon_qp = QuarticPolynomial(s0, c_speed, c_accel, tv, 0.0, Ti)

                tfp.s = [lon_qp.calc_point(t) for t in fp.t]
                tfp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
                tfp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
                tfp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]

                Jp = sum(np.power(tfp.d_ddd, 2))  # square of jerk
                Js = sum(np.power(tfp.s_ddd, 2))  # square of jerk

                # square of diff from target speed
                ds = (V - tfp.s_d[-1]) ** 2

                tfp.cd = K_J * Jp + K_T * Ti + K_D * tfp.d[-1] ** 2
                tfp.cv = K_J * Js + K_T * Ti + K_D * ds
                tfp.cf = K_LAT * tfp.cd + K_LON * tfp.cv

                frenet_paths_by_road_width[di].append(tfp)

    return frenet_paths_by_road_width

def calc_global_paths(fp_dict, csp):
    for di, fplist in fp_dict.items(): 
        for fp in fplist:
            # calc global positions
            for i in range(len(fp.s)):
                ix, iy = csp.calc_position(fp.s[i])
                if ix is None:
                    break
                i_yaw = csp.calc_yaw(fp.s[i])
                di = fp.d[i]
                fx = ix + di * math.cos(i_yaw + math.pi / 2.0)
                fy = iy + di * math.sin(i_yaw + math.pi / 2.0)
                fp.x.append(fx)
                fp.y.append(fy)
            # calc yaw and ds
            for i in range(len(fp.x) - 1):
                dx = fp.x[i + 1] - fp.x[i]
                dy = fp.y[i + 1] - fp.y[i]
                fp.yaw.append(math.atan2(dy, dx))
                fp.ds.append(math.hypot(dx, dy))
            if len(fp.yaw) == 1:
                return None
            else:
                fp.yaw.append(fp.yaw[-1])
                fp.ds.append(fp.ds[-1])

            # calc curvature
            for i in range(len(fp.yaw) - 1):
                fp.c.append((fp.yaw[i + 1] - fp.yaw[i]) / fp.ds[i])
            
    return fp_dict

def check_collision(fp, obs_paths):
    d = []
    for obs_path in obs_paths:
        for i in range(min(len(fp.x), len(obs_path.x))):
            dx = ((fp.x[i] - obs_path.x[i]) ** 2)
            dy = ((fp.y[i] - obs_path.y[i]) ** 2)
            d.append(dx + dy)

    collision= any([di <= (((0.5*ego_length)**2 + (0.5*ego_width)**2)+ ((0.5*obs_length)**2 + (0.5*obs_width)**2)) for di in d])
    
    if collision:
        return False

    return True
    # pass

def check_paths(fp_dict, obs_paths):
    if fp_dict is None:
        return None

    filtered_fp_dict = {}  # To store valid paths categorized by road width

    for di, fplist in fp_dict.items():  # Iterate over the dictionary
        ok_ind = []  # Indices of valid paths for the current road width

        for i, fp in enumerate(fplist):
            if any([v > MAX_EGO_SPEED for v in fp.s_d]):  # Max speed check
                continue
            elif any([abs(a) > MAX_EGO_ACCEL for a in fp.s_dd]):  # Max accel check
                continue
            elif any([abs(c) > MAX_CURVATURE for c in fp.c]):  # Max curvature check
                continue
            elif not check_collision(fp, obs_paths):  # Collision check
                continue

            ok_ind.append(i)

        if ok_ind:
            # Store only the valid paths for this road width
            filtered_fp_dict[di] = [fplist[i] for i in ok_ind]

    if not filtered_fp_dict:
        print("All candidate paths failed constraints!")
        return None

    return filtered_fp_dict

def frenet_optimal_planning(csp, s0, c_speed, c_accel, c_d, c_d_d, c_d_dd, obs_paths, V):
    fp_dict = calc_frenet_paths(c_speed, c_accel, c_d, c_d_d, c_d_dd, s0, V)
    fp_dict = calc_global_paths(fp_dict, csp)
    fp_dict = check_paths(fp_dict, obs_paths)
    
    if fp_dict is None:
        return None, None
    
    sorted_dict = {}
    min_cost = float("inf")
    best_path = None
    for di,fplist in fp_dict.items():
        sorted_dict[di] = sorted(fplist, key=lambda x: x.cf)
    
        # find minimum cost path  
        for fp in fplist:
            if min_cost >= fp.cf:
                min_cost = fp.cf
                best_path = fp
                
    return best_path, sorted_dict
    
def parameter(csp_1, s0, c_speed, c_accel, c_d, c_d_d, c_d_dd, obs_paths, Target_speed,lanes, obs_s0):
    if any ((20 > (obs_so - s0) >=0) and (lane == csp_1) for (obs_so,lane) in zip(obs_s0,lanes)):
        V1 = Target_speed + (20/3.6)
        path, fp_dict= frenet_optimal_planning(csp_1, s0, c_speed, c_accel, c_d, c_d_d, c_d_dd, obs_paths, V1)
    else:
        V2 = Target_speed
        path, fp_dict= frenet_optimal_planning(csp_1, s0, c_speed, c_accel, c_d, c_d_d, c_d_dd, obs_paths, V2)
    return path, fp_dict