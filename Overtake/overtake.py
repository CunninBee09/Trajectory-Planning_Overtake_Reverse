import numpy as np
import matplotlib.pyplot as plt
import copy
import math
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from FrenetOptimalTrajectory.frenet_optimal_trajectory import \
    frenet_optimal_planning

from CubicSplinePlanner import cubic_spline_planner

road_width = 10  #meters
obs_length = 5   #meters
obs_width = 2    #meters
dt = 0.2
SIM_LOOP = 500
MAX_SPEED = 50.0 / 3.6  # maximum speed [m/s]
MAX_ACCEL = 2.0  # maximum acceleration [m/ss]
MAX_CURVATURE = 1.0  # maximum curvature [1/m]
MAX_ROAD_WIDTH = 7.0  # maximum road width [m]
D_ROAD_W = 1.0  # road width sampling length [m]
DT = 0.2  # time tick [s]
MAX_T = 5.0  # max prediction time [m]
MIN_T = 4.0  # min prediction time [m]
TARGET_SPEED = 15.0 / 3.6  # target speed [m/s]
D_T_S = 5.0 / 3.6  # target speed sampling length [m/s]
N_S_SAMPLE = 1  # sampling number of target speed
ROBOT_RADIUS = 2.0  # robot radius [m]

# cost weights
K_J = 0.1
K_T = 0.1
K_D = 1.0       
K_LAT = 1.0 
K_LON = 1.0

show_animation = True


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


def calc_frenet_paths(c_speed, c_accel, c_d, c_d_d, c_d_dd, s0):
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
            for tv in np.arange(TARGET_SPEED - D_T_S * N_S_SAMPLE,
                                TARGET_SPEED + D_T_S * N_S_SAMPLE, D_T_S):
                tfp = copy.deepcopy(fp)
                lon_qp = QuarticPolynomial(s0, c_speed, c_accel, tv, 0.0, Ti)

                tfp.s = [lon_qp.calc_point(t) for t in fp.t]
                tfp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
                tfp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
                tfp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]

                Jp = sum(np.power(tfp.d_ddd, 2))  # square of jerk
                Js = sum(np.power(tfp.s_ddd, 2))  # square of jerk

                # square of diff from target speed
                ds = (TARGET_SPEED - tfp.s_d[-1]) ** 2

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

        fp.yaw.append(fp.yaw[-1])
        fp.ds.append(fp.ds[-1])

        # calc curvature
        for i in range(len(fp.yaw) - 1):
            fp.c.append((fp.yaw[i + 1] - fp.yaw[i]) / fp.ds[i])
            
        # for fp in fplist:
        #  plt.plot(fp.x, fp.y, '-g')  # Visualize all generated paths


    return fplist


# def check_collision(fp, obs_path ):
#     for i in range(len(obs_path.x)):
#         d = [((ix - obs_path.x[i]) ** 2 + (iy[i] - obs_path.y[i] ** 2))
#              for (ix, iy) in zip(fp.x, fp.y)]

#         collision = any([di <= ROBOT_RADIUS ** 2 for di in d])

#         if collision:
#             return False

#     return True

# def check_collision(fp, obs_path ):
#     for i in range(min(len(fp.x), len(obs_path.x))):
#         dx = ((fp.x[i] - obs_path.x[i]) ** 2)
#         dy = ((fp.y[i] - obs_path.y[i]) ** 2)
#         d = dx + dy

#         if ([d <= ROBOT_RADIUS *
# * 2] ): 
#             return False
#         break

#     return True


# def check_paths(fplist, obs_path):
#     ok_ind = []
#     for i, _ in enumerate(fplist):
#         if any([v > MAX_SPEED for v in fplist[i].s_d]):  # Max speed check
#             continue
#         elif any([abs(a) > MAX_ACCEL for a in
#                   fplist[i].s_dd]):  # Max accel check
#             continue
#         elif any([abs(c) > MAX_CURVATURE for c in
#                   fplist[i].c]):  # Max curvature check
#             continue
#         # elif not check_collision(fplist[i], obs_path):
#         #     continue

#         ok_ind.append(i)
        
#         if not ok_ind:
#          print("All candidate paths failed constraints!")  

#     return [fplist[i] for i in ok_ind]
    


def check_obs_paths(obslist):
    ok_ind = []
    for i, _ in enumerate(obslist):
        if any([v > MAX_SPEED for v in obslist[i].s_d]):  # Max speed check
            continue
        elif any([abs(a) > MAX_ACCEL for a in
                  obslist[i].s_dd]):  # Max accel check
            continue
        elif any([abs(c) > MAX_CURVATURE for c in
                  obslist[i].c]):  # Max curvature check
            continue
        # elif not check_collision(fplist[i], ob):
        #     continue

        ok_ind.append(i)

    return [obslist[i] for i in ok_ind]


# def frenet_optimal_planning(csp, s0, c_speed, c_accel, c_d, c_d_d, c_d_dd, obs_path):
#     fplist = calc_frenet_paths(c_speed, c_accel, c_d, c_d_d, c_d_dd, s0)
#     fplist = calc_global_paths(fplist, csp)
#     fplist = check_paths(fplist, obs_path)
    
#     # find minimum cost path
#     min_cost = float("inf")
#     best_path = None
#     for fp in fplist:
#         if min_cost >= fp.cf:
#             min_cost = fp.cf
#             best_path = fp
    
#     if not fplist:
#      print("No valid paths generated after checking constraints!")

#     return best_path

def obstacle_planning(csp, obs_s0, obs_speed, obs_acc, obs_d, obs_d_d, obs_d_dd):
    obslist= calc_frenet_paths(obs_speed, obs_acc, obs_d, obs_d_d, obs_d_dd, obs_s0)
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
    
def main():
    print(__file__ + " start!!")
    
    wx = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    wy = [0.0, 5.0, 10.0, 0.0, 0.0, 0.0]
    tx, ty, tyaw, tc, csp = generate_target_course(wx, wy)
    
    area = 20.0
    
 # initial state of obs vehicle
    obs_s0 = 10.0 # current position
    obs_speed = 15.0 / 3.6  # current speed [m/s]
    obs_acc= 0.0  # current acceleration [m/ss]
    obs_d = 0.0 #lateral positon[m]
    obs_d_d = 0.0 #lateral velocity[m/s]
    obs_d_dd = 0.0 #lateral acceleration[m/ss]
    
 # initial state of ego vehicle
    c_speed = 10.0 / 3.6  # current speed [m/s]
    c_accel = 0.0  # current acceleration [m/ss]
    c_d = 2.0  # current lateral position [m]
    c_d_d = 0.0  # current lateral speed [m/s]
    c_d_dd = 0.0  # current lateral acceleration [m/s]
    s0 = 0.0  # current course position

    for i in range(SIM_LOOP):
        
        obs_path = obstacle_planning(csp, obs_s0,obs_speed, obs_acc, obs_d, obs_d_d, obs_d_dd)
        # path = frenet_optimal_planning(csp, s0, c_speed, c_accel, c_d, c_d_d, c_d_dd, obs_path)
        
        obs_s0 = obs_path.s[1]
        obs_d = obs_path.d[1]
        obs_d_d = obs_path.d_d[1]
        obs_d_dd = obs_path.d_dd[1]
        obs_speed = obs_path.s_d[1]
        obs_acc = obs_path.s_dd[1]
        
        # path = frenet_optimal_planning(csp, s0, c_speed, c_accel, c_d, c_d_d, c_d_dd, obs_path)
        # if path is None:
        #     print("No valid path found for ego vehicle!")
        #     break
        
        # s0 = path.s[1]
        # c_d = path.d[1]
        # c_d_d = path.d_d[1]
        # c_d_dd = path.d_dd[1]
        # c_speed = path.s_d[1]
        # c_accel = path.s_dd[1]

        # if np.hypot(obs_path.x[1] - tx[-1], obs_path.y[1] - ty[-1]) <= 1.0:
        #     print("Goal")
        #     break

        if show_animation:  # pragma: no cover
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(tx, ty)
            plt.plot(obs_path.x[1:], obs_path.y[1:], "-or")
            plt.plot(obs_path.x[1], obs_path.y[1], "vc")
            plt.xlim(obs_path.x[1] - area, obs_path.x[1] + area)
            plt.ylim(obs_path.y[1] - area, obs_path.y[1] + area)
            # plt.plot(path.x[1:], path.y[1:], "-ob")
            # plt.plot(path.x[1], path.y[1], "vc")
            # plt.xlim(path.x[1] - area, path.x[1] + area)
            # plt.ylim(path.y[1] - area, path.y[1] + area)
            plt.title("v[km/h]:" + str(obs_speed * 3.6)[0:4])
            plt.grid(True)
            plt.pause(0.0001)


    print("Finish")
    if show_animation:  # pragma: no cover
        plt.grid(True)
        plt.pause(0.0001)
        plt.show()


if __name__ == '__main__':
    main()