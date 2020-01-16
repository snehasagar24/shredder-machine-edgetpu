import numpy as np
import cv2
import rtsp
cap = cv2.VideoCapture()

#cap = cv2.VideoCapture("rtsp://admin:Unilever@123@169.254.24.88")
cap.open('rtsp://admin:Unilever@123@169.254.24.88:80')


while(True):
     # Capture frame-by-frame
    ret, frame = cap.read()
    print(ret)
    print(frame.shape())

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
