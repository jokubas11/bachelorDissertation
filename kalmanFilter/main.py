from kalmanFilter import KalmanFilter
import numpy as np

"""Notice that these initialization values have been used
for simulation purposes, do not mean anything and are just
arbitrary values"""

# Initalization 

np.random.seed(0) # Defining random seed so results are reproducable
dt = 0.1
STEPS = 100
time = np.linspace(0, dt*STEPS, STEPS) # Defining time array

meas_variance = 2 ** 2; acceleration_variance = 0.25 ** 2
meas_value_x = 0.0; meas_value_y = 0.0

# Defining initial true state
real_pos_x = 0.0; real_pos_y = 0.0
initial_speed_x = 3.0; initial_speed_y = 3.0

# Defining initial estimates
initial_pos_x_est = 0.0; initial_pos_y_est = 0.0
initial_speed_x_est = 3.0; initial_speed_y_est = 3.0
no_of_state_variables = 4
no_of_state_measurements = 2

# Defining state Matrices x, P, F, Q, H, R
x = np.array([initial_pos_x_est, initial_speed_x_est,
              initial_pos_y_est, initial_speed_y_est])

P = np.diag([0, acceleration_variance, 0, acceleration_variance])

# F Defined by regular kinematics equations:
# x = x + dt * v
# v = v

F = np.array([[1, dt, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, dt],
              [0, 0, 0, 1]])

Q = np.diag([0, acceleration_variance, 0, acceleration_variance])
H = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]])

R = np.array([[meas_variance, 0],
              [0, meas_variance]])

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

# Initializing Kalman Filter
kf = KalmanFilter(initial_state = x,
                  initial_covariance = P,
                  process_model = F,
                  process_noise = Q,
                  measurement_function = H,
                  measurement_covariance = R,
                  no_of_state_variables = no_of_state_variables,
                  no_of_state_measurements = no_of_state_measurements)

step = 0 # step counter, used for simulation of abscent measurements only

# Loop that runs throughout the time array defined previously
for t in time:
    
    #Simulate real speed
    real_speed_x = initial_speed_x + np.random.randn() * np.sqrt(acceleration_variance)
    real_speed_y = initial_speed_y + np.random.randn() * np.sqrt(acceleration_variance)

    # Simulate real position
    real_pos_x = real_pos_x + dt * real_speed_x
    real_pos_y = real_pos_y + dt * real_speed_y
    
    # Simulate measurements with noise
    meas_value_x = real_pos_x + np.random.randn() * np.sqrt(meas_variance)
    meas_value_y = real_pos_y + np.random.randn() * np.sqrt(meas_variance)
    
    # Simulation of abscent measurements
    if step > 70 and step < 110:
        meas_value_x, meas_value_y = 0, 0
    
    # Assign obtained measurements to measurement array
    z = np.array([meas_value_x, meas_value_y])
    
    kf.predict(dt = dt)
    kf.update(state_measurement = z)
    step += 1 # increment step