from particleFilter import ParticleFilter
from detection import detector
import numpy as np
import cv2

N = 5000 # Number of particles, can be any number you like
initialPosition = np.array([12, 185]) # Initial known position
initialSpeed = np.array([0, 0]) # Initial known speed
initialStdDev = np.array([50, 50]) # Belief in position (Standard deviation)
initialConditions = np.array([initialPosition,
                              initialSpeed,
                              initialStdDev])

# Set of initial conditions
minimumDetectionArea = 2000
imageSize = (480, 360)
history_time = 50
image_no = 0
speed_mean = initialSpeed

# Creating an instance for VideoCapture object
# Your video file goes here
capture = cv2.VideoCapture("Video1.mp4") 

#creating an instance of BackgroundSubtractor object
bg_sub = cv2.createBackgroundSubtractorMOG2(history = history_time,
                                            detectShadows = False)

# Creating instance of ParticleFilter object
pf = ParticleFilter(N, initialConditions, imageSize)


#permanent loop while the frame is available
while(1):

    #reading the current frame
    ret, captured_frame = capture.read()

    #break the loop if no input is received
    if captured_frame is None:
        break

    #resizing the frame received
    captured_frame = cv2.resize(captured_frame, (imageSize))

    #applying background subtraction method to get foreground mask
    fg_mask = bg_sub.apply(captured_frame)

    #creating an instance of detector object
    DetectorObject = detector(fg_mask,
                              minimumDetectionArea, 
                              image_no,
                              captured_frame)

    pf.predict(speed_mean)
    centers, coordinates = DetectorObject.detect()
    tempCenters = centers
    
    if centers is not None:
        pf.update(centers)

    measureAndUpdateDifference = centers - tempCenters
    
    if pf.neff() < pf.N/2:
        pf.resample()
    
    pos_mean, pos_var, speed_mean, speed_var = pf.estimate()

    
    try:
        DetectorObject.draw_rectangles(coordinates - measureAndUpdateDifference, 
                                      (0, 0, 0))
    except:
        pass
    
    cv2.imshow('Detection on the Original Image', captured_frame)
    cv2.imshow('Detected Background After Filtering', fg_mask)
    cv2.waitKey(1)