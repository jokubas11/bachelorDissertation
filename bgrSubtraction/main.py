from detection import detector
import cv2

minimumDetectionArea = 2000
imageSize = (480, 360)
history_time = 50
image_no = 0

# Creating an instance for VideoCapture object
# Your video file goes here
capture = cv2.VideoCapture("Video1.mp4") 

#creating an instance of BackgroundSubtractor object
bg_sub = cv2.createBackgroundSubtractorMOG2(history = history_time,
                                            detectShadows = False)

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

    # Obtain measurements
    centers, coordinates = DetectorObject.detect()

    # Draws boxes where object detected, if no measurements, do nothing
    try:
        DetectorObject.draw_rectangles(coordinates, (0, 0, 0))
    except:
        pass
    
    cv2.imshow('Detection on the Original Image', captured_frame)
    cv2.imshow('Detected Background After Filtering', fg_mask)
    
    # Write images into desired folder
    # cv2.imwrite(f'pics\org_box\{image_no}.jpg', captured_frame)
    # cv2.imwrite(f'pics\c_bg_bf\{image_no}.jpg', fg_mask)
    
    #OpenCV syntax for display
    cv2.waitKey(1)