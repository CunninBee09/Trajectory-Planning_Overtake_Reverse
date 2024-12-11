road_width = 10  #meters
obs_length = 5   #meters
obs_width = 2    #meters
SIM_LOOP = 1000
MAX_EGO_SPEED = 60.0 / 3.6  # maximum speed of Ego vehicle [m/s]
MAX_EGO_ACCEL = 2.0  # maximum acceleration of Ego vehicle [m/ss]
MAX_OBS_SPEED = 15.0 / 3.6  # maximum speed of Obstacle [m/s]
MAX_OBS_ACCEL = 10.0  # maximum acceleration of Obstacle [m/ss]
MAX_CURVATURE = 10.0  # maximum curvature [1/m]
MAX_ROAD_WIDTH = 3.0  # maximum road width [m]
D_ROAD_W = 1 # road width sampling length [m]
DT = 0.2  # time tick [s]
MAX_T = 4.4  # max prediction time [m]
MIN_T = 4.0  # min prediction time [m]
TARGET_SPEED = 30.0 / 3.6  # target speed [m/s]
D_T_S = 5.0 / 3.6  # target speed sampling length [m/s]
N_S_SAMPLE = 4 # sampling number of target speed
ego_length = 5 #car_length [m]
ego_width = 2 #car_width [m]
ROBOT_RADIUS = 5.0# robot radius [m]

# cost weights
K_J = 0.1
K_T = 0.1
K_D = 1.5      
K_LAT = 1
K_LON = 0.5