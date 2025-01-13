import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import copy
import math
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from MultipleOvertake.Multi_ego_vehicle import parameter
from MultipleOvertake.Multi_obs_vehicle import obstacle_planning
from utilis import *
from CubicSplinePlanner import cubic_spline_planner
from matplotlib.patches import Rectangle

show_animation = True
#path 1
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
    
    #Horizontal Straight
    wx = [i for i in range(0, 301, 10)]  # x-coordinates from 0 to 200 in steps of 10
    wy = [0.0] * len(wx)  # y-coordinates remain constant (straight path)
    
    #Horizontal Curve
    # wx = [i for i in range(0, 301, 5)]
    # wy = [5 * np.sin(0.05 * x) for x in wx]
    
    #vertical straight
    # wy = [i for i in range(0, 301, 5)]
    # wx = [0.0] * len(wy)
    
    #Vertical Curve
    # wy = [i for i in range(0, 301, 5)]
    # wx = [5*np.sin(0.05*y) for y in wy]
    
    tx, ty, tyaw, tc, csp_1 = generate_target_course(wx, wy)
    
    #Way points for Lane 2
    ax = [x + 5.0*math.cos(i_yaw + math.pi / 2.0) for x,i_yaw in zip(tx,tyaw) ]
    ay = [y + 5.0*math.sin(i_yaw + math.pi / 2.0) for y,i_yaw in zip(ty,tyaw) ]
    
    px, py, pyaw, pc, csp_2 = generate_target_course(ax, ay)
    
    area = 30.0
    
    csp_1.ux = [x + 7.5*math.cos(i_yaw + math.pi / 2.0) for x,i_yaw in zip(tx,tyaw) ]
    csp_1.uy = [y + 7.5*math.sin(i_yaw + math.pi / 2.0) for y,i_yaw in zip(ty,tyaw) ]
    csp_1.lx = [x - 2.5*math.cos(i_yaw + math.pi / 2.0) for x,i_yaw in zip(tx,tyaw) ]
    csp_1.ly = [y - 2.5*math.sin(i_yaw + math.pi / 2.0) for y,i_yaw in zip(ty,tyaw) ]
    csp_1.mx = [x + 2.5*math.cos(i_yaw + math.pi / 2.0) for x,i_yaw in zip(tx,tyaw) ]
    csp_1.my = [y + 2.5*math.sin(i_yaw + math.pi / 2.0) for y,i_yaw in zip(ty,tyaw) ]
    
 # initial state of obs vehicle
    obs_s0 = [50.0, 100.0, 150.0] # current position
    Target_obs_speed = float(input("Enter obs speed in kmph:"))/3.6
    obs_speed =  [Target_obs_speed, Target_obs_speed, Target_obs_speed] # current speed [m/s]
    obs_acc= [0.0, 0.0, 0.0]  # current acceleration [m/ss]
    obs_d = [0.0, 0.0, 0.0] #lateral positon[m]
    obs_d_d = [0.0, 0.0, 0.0] #lateral velocity[m/s]
    obs_d_dd = [0.0, 0.0, 0.0] #lateral acceleration[m/ss]
    
 # initial state of ego vehicle
    Target_speed = float(input("Enter ego speed in kmph:"))/3.6
    c_speed = Target_speed # current speed [m/s]
    c_accel = 0.0  # current acceleration [m/ss]
    c_d = 0.0  # current lateral position [m]
    c_d_d = 0.0  # current lateral speed [m/s]
    c_d_dd = 0.0  # current lateral acceleration [m/s]
    s0 = 0.0  # current course position
    
    for i in range(SIM_LOOP):
        obs_paths=[]
        lanes = []
        for j in range(len(obs_s0)) :
            if j == 1:
                lane = csp_2
            else:
                lane = csp_1
            obs_path = obstacle_planning(lane, obs_s0[j], obs_speed[j], obs_acc[j], obs_d[j], obs_d_d[j], obs_d_dd[j],Target_obs_speed)
            lanes.append(lane)
            obs_paths.append(obs_path)
        
        if np.hypot(obs_path.x[1] - tx[-1], obs_path.y[1]-ty[-1]) <=3.0:
            print("Obstacle reached Goal first")
            break
        
        path,fp_dict = parameter(csp_1, s0, c_speed, c_accel, c_d, c_d_d, c_d_dd, obs_paths, Target_speed,lanes, obs_s0)
                
        if path is None:
            print("No valid path found for ego vehicle!")
            break
        
        if np.hypot(path.x[1] - tx[-1], path.y[1] - ty[-1]) <= 0.0:
            print("Goal")
            break
        
        #postion update
        for j in range (len(obs_s0)):
            obs_s0[j] = obs_paths[j].s[1]
            obs_d[j] = obs_paths[j].d[1]
            obs_d_d[j] = obs_paths[j].d_d[1]
            obs_d_dd[j] = obs_paths[j].d_dd[1]
            obs_speed[j] = obs_paths[j].s_d[1]
            obs_acc[j] = obs_paths[j].s_dd[1]
        
        #Updating Current Values of Ego_Vehicle
        s0 = path.s[1]
        c_d = path.d[1]
        c_d_d = path.d_d[1]
        c_d_dd = path.d_dd[1]
        c_speed = path.s_d[1]
        c_accel = path.s_dd[1]
        
        if show_animation:  # pragma: no cover
            plt.cla()
            plt.axis('equal')
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
            for di, fplist in fp_dict.items():
                if fplist is None:
                    continue# Only plot the top 10 paths
                else:
                    for i,fp in enumerate(fplist[:1]):
                        plt.plot(fp.x, fp.y, label=f"Path {i + 1} (Cost: {fp.cf:.4f})", linestyle="--", color = "blue", alpha = 0.5)
               
            # Draw obstacle vehicle as a rectangle
            for obs_path in obs_paths :
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
            
            #Plotting
            plt.figure(1)
            plt.plot(tx, ty)
            plt.plot(px,py)
            plt.plot(csp_1.ux, csp_1.uy, '-k')
            plt.plot(csp_1.lx, csp_1.ly, '-k')
            plt.plot(csp_1.mx, csp_1.my, '--k')
            for obs_path in obs_paths:
                # print("obs_s0 position is",obs_s0)
                # print("obs_path values are",obs_path.x[1], obs_path.y[1])
                plt.plot(obs_path.x[1:], obs_path.y[1:], "-r")
                plt.plot(obs_path.x[1], obs_path.y[1], "vc")
                plt.xlim(obs_path.x[1] - area, obs_path.x[1] + area)
                plt.ylim(obs_path.y[1] - area, obs_path.y[1] + area)
            # print("path values are",path.x[1], path.y[1])
            # print("s0 position is",s0)
            plt.plot(path.x[1:], path.y[1:], "-r")
            plt.plot(path.x[1], path.y[1], "vc")
            plt.xlim(path.x[1] - area, path.x[1] + area)
            plt.ylim(path.y[1] - area, path.y[1] + area)
            plt.title(f"Ego v[km/h]: {c_speed * 3.6:.2f}, Obs v[km/h]: {Target_obs_speed * 3.6:.2f}")         
            plt.grid(True)
            plt.pause(0.0001)

    print("Finish")
    if show_animation:  # pragma: no cover
        plt.grid(True)
        plt.pause(0.0001)
        plt.show()

if __name__ == '__main__':
    main()