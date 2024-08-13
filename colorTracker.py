import cv2
import numpy as np

video = cv2.VideoCapture(0)
# List to store the center points of the object
object_path = []

while (1):

    ret, frame = video.read()
    #mirror the image
    frame = cv2.flip(frame, 1)

    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 

    lower_h=0;
    lower_s=158;
    lower_v=142;
    upper_h=179;
    upper_s=255;
    upper_v=255;


    # Create the mask
    lower_red = np.array([lower_h, lower_s, lower_v])
    upper_red = np.array([upper_h, upper_s, upper_v])
    red_mask = cv2.inRange(hsvFrame, lower_red, upper_red)

    # Bitwise-AND mask and original image
    #red_res = cv2.bitwise_and(frame, frame, mask=red_mask)

    # # Display the mask, and the result
    # cv2.imshow('Original Frame', frame)
    # cv2.imshow('Red Mask', red_mask)
    # cv2.imshow('Red Detection', red_res)

    # Find contours in the mask
    contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original frame
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

    # Find the centroid of the largest contour
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        moments = cv2.moments(largest_contour)
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            # Add the centroid to the path
            object_path.append((cx, cy))

    # Draw the path of the object
    for i in range(1, len(object_path)):
        cv2.line(frame, object_path[i - 1], object_path[i], (255, 0, 0), 2)

    # Display the original frame with contours and path
    cv2.imshow('Original Frame with Path', frame)

    # kernel = np.ones((5, 5), "uint8") 

    #  # Dilate mask to fill in gaps #büyütüyor gibi düşünebiliriz
    # red_mask = cv2.dilate(red_mask, kernel)

    # #bitwise AND operation between the original frame and itself, This isolates the red regions in the original frame
    # res_red = cv2.bitwise_and(frame, frame, mask=red_mask)

    # # Find contours for red color
    # contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # for contour in contours:
    #     area = cv2.contourArea(contour)
    #     if area > 300:
    #         x, y, w, h = cv2.boundingRect(contour)
    #         frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

  
    #cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q') :
        break

video.release()
cv2.destroyAllWindows()