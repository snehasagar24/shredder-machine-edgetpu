import numpy as np
import cv2
import rtsp

#cap = cv2.VideoCapture("rtsp://admin:Unilever@123@169.254.24.88")
#cap = cv2.VideoCapture('rtsp://169.254.24.88')

while(True):
    with rtsp.Client(rtsp_server_uri = 'rtsp://169.254.24.88') as client:
        print(client)
        frame = client.read()
        frame = np.array(frame)
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