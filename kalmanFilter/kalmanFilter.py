import numpy as np

class KalmanFilter(object):
    
    def __init__(self, initial_state, initial_covariance, process_model,
                 process_noise, measurement_function, measurement_covariance,
                 no_of_state_variables, no_of_state_measurements):
    
        self.x = initial_state
        self.P = initial_covariance
        self.F = process_model
        self.Q = process_noise
        self.H = measurement_function
        self.R = measurement_covariance
        self.dim_x = no_of_state_variables
        self.dim_z = no_of_state_measurements
        
    def predict(self, dt):
    
        # KALMAN FILTER EQUATIONS:
        # x = F * x
        # P = F * P * F^T + Q, where ^T is equivalent to transposing matrix
        
        self.x = self.F.dot(self.x)
        self.P = self.F.dot(self.P).dot(self.F.T) + self.Q
    
    def update(self, state_measurement):
        
        z = state_measurement
    
        # Handling case where no measurements are received
        if np.all(z == np.array([0,0])):
            z = np.array([self.x[0], self.x[2]])

        # y = z - H * x
        # S = H * P * H^T + R
        # K = P * H^T
        # x = x + K * y
        # P = (I - K * H) * P

        # Here I is identity matrix and ^T is matrix transpose operation

        y = z - self.H.dot(self.x)
        S = self.H.dot(self.P).dot(self.H.T) + self.R
        K = self.P.dot(self.H.T).dot(np.linalg.inv(S))
        
        self.x = self.x + K.dot(y)
        self.P = (np.eye(self.dim_x) - K.dot(self.H)).dot(self.P)
    
    # Properties used for debugging purposes only
    @property
    def covs(self) -> np.array:
        return self.P
    
    @property
    def mean(self) -> np.array:
        return self.x