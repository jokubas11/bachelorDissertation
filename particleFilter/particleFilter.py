import numpy as np
np.random.seed(0)

class ParticleFilter(object):
    
    def __init__(self, N, initialConditions, imageSize):
        
        '''__init__ constructor is a special python function that is
        called automatically every time when memory is allocated
        for a new object. Here we initialize object attributes and
        call class methods that create particles and weights for
        those particles.'''
        
        self.imageSize = imageSize
        self.N = N
        self.initialConditions = initialConditions
        
        if self.initialConditions is not None:
            self.initialCoordinates = np.array([self.initialConditions[0, 0],
                                                self.initialConditions[0, 1]])
            self.initialSpeed = np.array([self.initialConditions[1, 0],
                                        self.initialConditions[1, 1]])
            self.initialStdDev = np.array([self.initialConditions[2, 0],
                                        self.initialConditions[2, 1]])
        
        self.create_particles()
        self.create_weights()
    
    def create_particles(self):
        
        '''If we know about the initial state of the system
        we create normally distributed particles around that
        state, if we do not know anything, we create uniform
        particles throughout.'''
        
        if self.initialConditions is None:
            self.create_uniform_particles()
        else:
            self.create_gaussian_particles()
        
        '''Particles 0 and 1 stand for position in x and y, and particles
        2 and 3 stand for velocity in x and y directions.'''
    
    def create_gaussian_particles(self):
        
        self.particles = np.empty((self.N, 4))
        
        self.particles[:, 0] = self.initialCoordinates[0] + \
                               (np.random.randn(self.N) * self.initialStdDev[0])
        
        self.particles[:, 1] = self.initialCoordinates[1] + \
                               (np.random.randn(self.N) * self.initialStdDev[1])
        
        self.particles[:, 2] = self.initialSpeed[0] + \
                               (np.random.randn(self.N) * self.initialStdDev[0])
        
        self.particles[:, 3] = self.initialSpeed[1] + \
                               (np.random.randn(self.N) * self.initialStdDev[1])
    
    def create_uniform_particles(self):
        self.particles = np.empty((self.N, 4))
        
        self.particles[:, 0] = np.random.uniform(0, self.imageSize[0], self.N)
        self.particles[:, 1] = np.random.uniform(0, self.imageSize[1], self.N)
        
        # -10 and 10 are trivial values for limits of speed particles
        self.particles[:, 2] = np.random.uniform(-10, 10, self.N)
        self.particles[:, 3] = np.random.uniform(-10, 10, self.N)

    # create weights for initial particles
    def create_weights(self):
        self.weights = np.ones(self.N) / self.N

    # find number of effective particles
    def neff(self):
        return 1 / np.sum(np.square(self.weights))

    # find mean and variance of weighted particles
    def estimate(self):
        particles_position = self.particles[:, 0:2]
        particles_speed = self.particles[:, 2:4]
        position_mean,position_var = self.mean_and_variance(particles_position)
        speed_mean, speed_var = self.mean_and_variance(particles_speed)
        return position_mean, position_var, speed_mean, speed_var
    
    def mean_and_variance(self, the_array):
        mean = np.average(the_array, weights=self.weights, axis=0)
        var = np.average((the_array - mean)**2, weights=self.weights, axis=0)
        return mean, var
    
    # We move particles according to how we think they move
    def predict(self, speed):
        self.particles[:, 0] += speed[0] + (np.random.randn(self.N) * 10)
        self.particles[:, 1] += speed[1] + (np.random.randn(self.N) * 10)
        self.particles[:, 2] += (np.random.randn(self.N) * 0.1)
        self.particles[:, 3] += (np.random.randn(self.N) * 0.1)
    
    def update(self, measurements):
        # Find how far are distance particles from measurement
        distance_x = np.abs(self.particles[:, 0] - measurements[0])
        distance_y = np.abs(self.particles[:, 1] - measurements[1])
        distance = self.euclidean_distance(distance_x, distance_y)
        
        # Find how far are speed particles from measurements
        # As measurement for speed we use difference of previous and current position
        speed_x = np.abs(self.particles[:, 2] - measurements[2])
        speed_y = np.abs(self.particles[:, 3] - measurements[3])
        speed = self.euclidean_distance(speed_x, speed_y)
        
        # Combine them
        total_difference = distance + speed
        
        # If measurement is far away from particle, it means its weight is low
        new_weights = 1/total_difference
        
        # Implementing Bayes Theorem
        self.weights *= new_weights
        self.weights += 1.e-50 # avoid round-off to zero
        self.weights /= sum(self.weights) # normalize
    
    # Function that calculates distance of particle from measurement
    def euclidean_distance(self, dist_x, dist_y):
        return np.sqrt(dist_x ** 2 + dist_y ** 2)

    # Resampling technique, see dissertation
    def resample(self):
        # Obtain indexes of particles we want to keep
        indexes = self.return_indexes()
        
        # Make new particles from those indexes
        self.resampled_particles(indexes)

    def return_indexes(self):
        sampling_positions = (np.arange(self.N) + \
                              np.random.uniform(0, 1/self.N)) / self.N
        
        indexes = np.zeros(self.N, 'i')
        cumulative_sum = np.cumsum(self.weights)
        
        i, j = 0, 0
        while i < self.N:
            if sampling_positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        return indexes

    def resampled_particles(self, indexes):
        resampled_weights = []
        resampled_particles = np.empty((self.N, 4))
        
        j = 0
        for i in indexes:
            resampled_weights.append(self.weights[i])
            resampled_particles[j, 0:4] = self.particles[i, 0:4]
            j += 1
        
        resampled_weights /= sum(resampled_weights)
        self.weights = resampled_weights
        self.particles = resampled_particles