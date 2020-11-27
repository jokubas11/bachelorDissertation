import cv2
import numpy as np

"""Creating object detector class with a detect method"""
class detector(object):

    # Creating the constructor of detector class
    def __init__(self, frame, minimumDetectionArea, image_no, captured_frame):
        self.frame = frame
        self.minimumDetectionArea = minimumDetectionArea
        self.image_no = image_no
        self.captured_frame = captured_frame

    # This function is for rectangle drawing on the picture
    def draw_rectangles(self, coordinates, color):
        for i in coordinates:
            cv2.rectangle(self.captured_frame, (i[0], i[1]), (i[2], i[3]), \
            color, 2)

    def detect(self):
        '''Centers are measurements we obtain for filtering, i.e. point
        tracking. Coordiantes are trivial, for rectangle drawing
        on a picture, not actually used for tracking.
        Here we find countours of every shape, find center of
        this shape, and if it passes the minimumDetectionArea,
        it is then counted as a detected human.'''
    
        centers, coordinates = [], []
        self.frame = cv2.medianBlur(self.frame, 3) # Median Filter

        # Find Countours
        contours, hierarchy = cv2.findContours(self.frame, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
        boundingRectangles = [None]*len(contours)

        # Find Coordinates of every bounding rectangle of every countour
        for i in range(len(contours)):
            boundingRectangles[i] = cv2.boundingRect(contours[i])
        
        for i in range(len(boundingRectangles)):
            x1 = (int(boundingRectangles[i][0]))
            y1 = (int(boundingRectangles[i][1]))
            x2 = (int(boundingRectangles[i][0]+boundingRectangles[i][2]))
            y2 = (int(boundingRectangles[i][1]+boundingRectangles[i][3]))
            center = [((x2 + x1) / 2), ((y2 + y1) / 2)]

            # Apply Threshold
            if ((abs(x2 - x1) * abs(y2 - y1)) > self.minimumDetectionArea):
                centers.append(center)
                coordinates.append(np.array([x1, y1, x2, y2]))
            
        if len(centers) == 0:
            return None, None

        return centers, coordinates