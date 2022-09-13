import cv2
import time
import numpy as np

# save output as ouput.avi
fourcc = cv2.VideoWriter_fourcc(*"XVID")
outputfile = cv2.VideoWriter("output.avi", fourcc, 20.0, (640, 480))

# starting webcam :)
cap = cv2.VideoCapture(0)

# Allowing the webcam to start by making the code sleep for 2 seconds
time.sleep(2)
bg = 0

# Capturing background for 60 frames
for i in range(60):
    ret, bg = cap.read()

# Flipping the background
bg = np.flip(bg, axis=1)

# Reading the captured frame until the camera is open
while(cap.isOpened()):
    ret, img = cap.read()
    if not ret:
        break
    img = np.flip(img, axis=1)

    # Converting the color from BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Generating mask to detect red colour
    # These values can also be changed as per the color
    lower_green = np.array([52, 0, 55])
    upper_green = np.array([104, 255, 255])
    mask_1 = cv2.inRange(hsv, lower_green, upper_green)

    lower_green = np.array([36, 25, 25])
    upper_green = np.array([70, 255, 255])
    mask_2 = cv2.inRange(hsv, lower_green, upper_green)
    # mask
    mask_1 = mask_1 + mask_2

    # Open and expand the image where there is mask 1 (color)
    mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_OPEN,
                              np.ones((3, 3), np.uint8))
    mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_DILATE,
                              np.ones((3, 3), np.uint8))

    # Selecting only the part that does not have mask 1 and saving in mask 2
    mask_2 = cv2.bitwise_not(mask_1)
    # Keeping only the part of the images without the red color
    res_1 = cv2.bitwise_and(img, img, mask=mask_2)
    # Keeping only the part of the images with the red color
    res_2 = cv2.bitwise_and(bg, bg, mask=mask_1)

    # Generating the final output by merging res_1 and res_2
    final_output = cv2.addWeighted(res_1, 1, res_2, 1, 0)
    outputfile.write(final_output)
    # Displaying the output to the user
    cv2.imshow("magic", final_output)
    cv2.waitKey(1)

cap.release()

cv2.destroyAllWindows()
# code completed :)
