import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import copy
import math
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from ego_vehicle import parameter
from obs_vehicle import obstacle_planning
from utilis import *
from CubicSplinePlanner import cubic_spline_planner
from matplotlib.patches import Rectangle

show_animation = True

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
    
    wx = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0]
    wy = [0.0, 10.0, 10.0, 5.0, 0.0, -5.0, -5.0, 0.0, 2.0, 5.0, 7.0, 10.0, 5.0, 3.0, -1.0, -4.0]
    # wx = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0]
    # wy = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    tx, ty, tyaw, tc, csp = generate_target_course(wx, wy)
    
    area = 50.0
    
    csp.ux = [x + 7.5*math.cos(i_yaw + math.pi / 2.0) for x,i_yaw in zip(tx,tyaw) ]
    csp.uy = [y + 7.5*math.sin(i_yaw + math.pi / 2.0) for y,i_yaw in zip(ty,tyaw) ]
    csp.lx = [x - 2.5*math.cos(i_yaw + math.pi / 2.0) for x,i_yaw in zip(tx,tyaw) ]
    csp.ly = [y - 2.5*math.sin(i_yaw + math.pi / 2.0) for y,i_yaw in zip(ty,tyaw) ]
    csp.mx = [x + 2.5*math.cos(i_yaw + math.pi / 2.0) for x,i_yaw in zip(tx,tyaw) ]
    csp.my = [y + 2.5*math.sin(i_yaw + math.pi / 2.0) for y,i_yaw in zip(ty,tyaw) ]
    
 # initial state of obs vehicle
    obs_s0 = 40.0 # current position
    Target_obs_speed = float(input("Enter obs speed in kmph:"))/3.6
    obs_speed =  Target_obs_speed # current speed [m/s]
    obs_acc= 0.0  # current acceleration [m/ss]
    obs_d = 0.0 #lateral positon[m]
    obs_d_d = 0.0 #lateral velocity[m/s]
    obs_d_dd = 0.0 #lateral acceleration[m/ss]
    
 # initial state of ego vehicle
    Target_speed = float(input("Enter ego speed in kmph:"))/3.6
    c_speed = Target_speed # current speed [m/s]
    c_accel = 0.0  # current acceleration [m/ss]
    c_d = 0.0  # current lateral position [m]
    c_d_d = 0.0  # current lateral speed [m/s]
    c_d_dd = 0.0  # current lateral acceleration [m/s]
    s0 = 0.0  # current course position
    
    for i in range(SIM_LOOP):
        
        obs_path = obstacle_planning(csp, obs_s0, obs_speed, obs_acc, obs_d, obs_d_d, obs_d_dd,Target_obs_speed)
        
        if np.hypot(obs_path.x[1] - tx[-1], obs_path.y[1]-ty[-1]) <=10.0:
            print("Obstacle reached Goal first")
            break
        
        path,fplist = parameter(csp, s0, c_speed, c_accel, c_d, c_d_d, c_d_dd, obs_path, Target_speed)
        # path, fplist = frenet_optimal_planning(csp, s0, c_speed, c_accel, c_d, c_d_d, c_d_dd, obs_path, csp.uy ,csp.ly)
        
        
        if path is None:
            break
        
        if np.hypot(path.x[1] - tx[-1], path.y[1] - ty[-1]) <= 0.0:
            print("Goal")
            break
        
        
        
        obs_s0 = obs_path.s[1]
        obs_d = obs_path.d[1]
        obs_d_d = obs_path.d_d[1]
        obs_d_dd = obs_path.d_dd[1]
        obs_speed = obs_path.s_d[1]
        obs_acc = obs_path.s_dd[1]
        
        # path = frenet_optimal_planning(csp, s0, c_speed, c_accel, c_d, c_d_d, c_d_dd, obs_path, uy)
        # if path is None:
        #     print("No valid path found for ego vehicle!")
        #     break
        
        s0 = path.s[1]
        c_d = path.d[1]
        c_d_d = path.d_d[1]
        c_d_dd = path.d_dd[1]
        c_speed = path.s_d[1]
        c_accel = path.s_dd[1]
        
        # plt.figure()
        # for i, path in enumerate(fplist[:10]):
        #   plt.plot(path.x, path.y, label=f"Path {i} (Cost: {path.cf:.2f})")


        if show_animation:  # pragma: no cover
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
             
            #Draw ego vehicle as rectangle
            ego_yaw = path.yaw[1] * 180 / np.pi  # Convert from radians to degrees for matplotlib
            ego_vehicle = Rectangle(
             (path.x[1] - ego_length / 2, path.y[1] - ego_width / 2),  # Bottom-left corner
              ego_length,  # Length
              ego_width,   # Width
            #   angle = ego_yaw,
              edgecolor="blue",
              facecolor="none"
            )
            
            # Apply rotation around the center of the rectangle
            t = transforms.Affine2D().rotate_deg_around(path.x[1], path.y[1], ego_yaw) + plt.gca().transData
            ego_vehicle.set_transform(t)

            plt.gca().add_patch(ego_vehicle)
            
            # Plot the top 10 paths
            for i, fp in enumerate(fplist[:10]):  # Only plot the top 10 paths
               plt.plot(fp.x, fp.y, label=f"Path {i + 1} (Cost: {fp.cf:.4f})", linestyle="--")
               
            # Draw obstacle vehicle as a rectangle
            obs_yaw = obs_path.yaw[1] * 180 / np.pi
            obs_vehicle = Rectangle(
             (obs_path.x[1] - obs_length / 2, obs_path.y[1] - obs_width / 2),  # Bottom-left corner
              obs_length,  # Length
              obs_width,   # Width
              edgecolor="red",
              facecolor="none"
            )
            t = transforms.Affine2D().rotate_deg_around(obs_path.x[1], obs_path.y[1], obs_yaw) + plt.gca().transData
            obs_vehicle.set_transform(t)
            
            plt.gca().add_patch(obs_vehicle)
            
            
            plt.plot(tx, ty)
            plt.plot(csp.ux, csp.uy, '-k')
            plt.plot(csp.lx, csp.ly, '-k')
            plt.plot(csp.mx, csp.my, '--k')
            plt.plot(obs_path.x[1:], obs_path.y[1:], "-r")
            plt.plot(obs_path.x[1], obs_path.y[1], "vc")
            plt.xlim(obs_path.x[1] - area, obs_path.x[1] + area)
            plt.ylim(obs_path.y[1] - area, obs_path.y[1] + area)
            plt.plot(path.x[1:], path.y[1:], "-r")
            plt.plot(path.x[1], path.y[1], "vc")
            plt.xlim(path.x[1] - area, path.x[1] + area)
            plt.ylim(path.y[1] - area, path.y[1] + area)
            plt.title(f"Ego v[km/h]: {c_speed * 3.6:.2f}, Obs v[km/h]: {obs_speed * 3.6:.2f}")
            plt.grid(True)
            plt.pause(0.0001)


    print("Finish")
    if show_animation:  # pragma: no cover
        plt.grid(True)
        plt.pause(0.0001)
        plt.show()


if __name__ == '__main__':
    main()