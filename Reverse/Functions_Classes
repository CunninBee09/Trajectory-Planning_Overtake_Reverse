# File containing all the functions and classes 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import copy
import math
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from QuinticPolynomialsPlanner.quintic_polynomials_planner import \
    QuinticPolynomial
from QuinticPolynomialsPlanner.quintic_polynomials_planner import quintic_polynomials_planner
from CubicSpline import cubic_spline_planner
from matplotlib.patches import Rectangle

SIM_LOOP = 1000

# Parameter
MAX_SPEED = 50.0 / 3.6  # maximum speed [m/s]
MAX_ACCEL = 2.0  # maximum acceleration [m/ss]
MAX_CURVATURE = 1.0  # maximum curvature [1/m]
MAX_ROAD_WIDTH = 5.0  # maximum road width [m]
D_ROAD_W = 1.0  # road width sampling length [m]
DT = 0.2  # time tick [s]
MAX_T = 5.0  # max prediction time [m]
MIN_T = 4.0  # min prediction time [m]
Target_speed = 20.0 / 3.6  # target speed [m/s]
D_T_S = 4.0 / 3.6  # target speed sampling length [m/s]
N_S_SAMPLE = 6  # sampling number of target speed
ego_length = 5 #car_length [m]
ego_width = 2 #car_width [m]
#Parking Space Parameters 

parkx = 40.0 # x coordinate for the parking 
parky = -10.0 # y coordinate for the parking 
pspot_length = 6.0
psport_breadth = 4.0
# cost weights
K_J = 0.1                                                                                       
K_T = 0.1    
K_D = 0.2   
K_LAT = 1.5    
K_LON = 1.2   

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
        
def calc_frenet_paths(c_speed, c_accel, c_d, c_d_d, c_d_dd, s0,V):
    frenet_paths = []

    # generate path to each offset goal
    for di in np.arange(-MAX_ROAD_WIDTH, MAX_ROAD_WIDTH, D_ROAD_W):

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
            for tv in np.arange(V - D_T_S * N_S_SAMPLE,
                                V + D_T_S * N_S_SAMPLE, D_T_S):
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

                frenet_paths.append(tfp)

    return frenet_paths

def calc_global_paths(fplist, csp):
    for fp in fplist:

        # calc global positions
        for i in range(len(fp.s)):
            ix, iy = csp.calc_position(fp.s[i])
            if ix is None:
                continue
            i_yaw = csp.calc_yaw(fp.s[i])
            di = fp.d[i]
            fx = ix + di * math.cos(i_yaw + math.pi / 2.0) 
            fy = iy + di * math.sin(i_yaw + math.pi / 2.0) 
            fp.x.append(fx)
            fp.y.append(fy)

        # calc yaw and ds
        for i in range(len(fp.x) - 1):
            dx = fp.x[i + 1] - fp.x[i]       #+1
            dy = fp.y[i + 1] - fp.y[i]       #+1
            fp.yaw.append(math.atan2(dy, dx))
            fp.ds.append(math.hypot(dx, dy))

        
        fp.yaw.append(fp.yaw[-1])
        fp.ds.append(fp.ds[-1])

    return fplist

def check_paths(fplist):
    ok_ind = []
    for i, _ in enumerate(fplist):
    
        if any([v > MAX_SPEED for v in fplist[i].s_d]):  # Max speed check
            continue
        elif any([abs(a) > MAX_ACCEL for a in
                  fplist[i].s_dd]):  # Max accel check
            continue
        elif any([abs(c) > MAX_CURVATURE for c in
                  fplist[i].c]):  # Max curvature check
            continue

        ok_ind.append(i)

    return [fplist[i] for i in ok_ind]

def frenet_optimal_planning(csp, s0, c_speed, c_accel, c_d, c_d_d, c_d_dd,V):
    fplist = calc_frenet_paths(c_speed, c_accel, c_d, c_d_d, c_d_dd, s0,V)
    fplist = calc_global_paths(fplist, csp)
    fplist = check_paths(fplist)
    
    fplist.sort(key=lambda x: x.cf)
    # find minimum cost path
    min_cost = float("inf")
    best_path = None
    for fp in fplist:
        if min_cost >= fp.cf:
            min_cost = fp.cf
            best_path = fp

    return best_path,fplist

def generate_target_course(x, y):
    csp = cubic_spline_planner.CubicSpline2D(x, y) 
    s = np.arange(0, csp.s[-1], 0.1)

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = csp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(csp.calc_yaw(i_s))
        rk.append(csp.calc_curvature(i_s))

    return rx, ry, ryaw, rk, csp

def generate_reverse_target_course(x, y):
    csp = cubic_spline_planner.CubicSpline2D(x, y) 
    s = np.arange(0, csp.s[-1], 0.1)

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = csp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(csp.calc_yaw(i_s))
        rk.append(csp.calc_curvature(i_s))

    return rx, ry, ryaw, rk, csp


def generate_s_trajectory(x0, y0, parkx, parky,parking_spot_length, parking_spot_breath, m0=1, mf=0, k0=0, kf=0, num_points=40):
   
    # Constructing a system of equations to solve for polynomial coefficients
    xf = parkx + parking_spot_length/2.0
    yf = parky
    A = np.array([
        [x0**5, x0**4, x0**3, x0**2, x0, 1],
        [xf**5, xf**4, xf**3, xf**2, xf, 1],
        [5*x0**4, 4*x0**3, 3*x0**2, 2*x0, 1, 0],
        [5*xf**4, 4*xf**3, 3*xf**2, 2*xf, 1, 0],
        [20*x0**3, 12*x0**2, 6*x0, 2, 0, 0],
        [20*xf**3, 12*xf**2, 6*xf, 2, 0, 0]
    ])
    b = np.array([y0, yf, m0, mf, k0, kf])

    # Solve for coefficients
    coefficients = np.linalg.solve(A, b)

    # Generate x and y points
    x = np.linspace(x0, xf, num_points)
    y = np.polyval(coefficients, x)
    return x,y

def straight_into_reverse(parking_x, parking_y, parking_spot_length, parking_spot_breath):
    startx = parking_x + parking_spot_length/2.0
    endx = parking_x - parking_spot_length/2.0
    starty = endy = parking_y
    straight_wx = []
    straight_wy = []

    r = np.arange(startx,endx, -0.1)
    for i in r:
        straight_wx.append(i)
        straight_wy.append(starty)
    
    return straight_wx,straight_wy

def check_for_transition(csp, s0, c_speed, c_accel, c_d, c_d_d, c_d_dd, parkx, parky, Target_speed):
    if s0 < parkx :
        
        V1 = Target_speed
        
        path, fplist= frenet_optimal_planning(csp, s0, c_speed, c_accel, c_d, c_d_d, c_d_dd,V1)
    
    elif s0 > parkx and s0 < parkx +20.0:
        
        V2 = 0.0/3.6

        path, fplist= frenet_optimal_planning(csp, s0, c_speed, c_accel, c_d, c_d_d, c_d_dd,V2)


    return path, fplist


def forward_parking(csp, s0, c_speed, c_accel, c_d, c_d_d, c_d_dd, parkx, parky, target_speed):
    path, fplist = check_for_transition(csp, s0, c_speed, c_accel, c_d, c_d_d, c_d_dd, parkx, parky, target_speed)
    s0 = path.s[1]
    c_d = path.d[1]
    c_d_d = path.d_d[1]
    c_d_dd = path.d_dd[1]
    c_speed = path.s_d[1]
    c_accel = path.s_dd[1]
    ego_x = path.x[1]
    ego_y = path.y[1]

    return s0, c_d, c_d_d, c_d_dd, c_speed, c_accel, path, ego_x, ego_y, fplist

def reverse_parking(csp, rs0, r_speed, r_accel, r_d, r_d_d, r_d_dd, parkx, parky, target_speed):
    path, fplist = check_for_transition(csp, rs0, r_speed, r_accel, r_d, r_d_d, r_d_dd, parkx, parky, target_speed)
    rs0 = path.s[1]
    r_d = path.d[1]
    r_d_d = path.d_d[1]
    r_d_dd = path.d_dd[1]
    r_speed = path.s_d[1]
    r_accel = path.s_dd[1]
    ego_x = path.x[1]
    ego_y = path.y[1]

    return rs0, r_d, r_d_d, r_d_dd, r_speed, r_accel, path, ego_x,ego_y,fplist       
