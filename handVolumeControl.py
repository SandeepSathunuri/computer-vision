import cv2
import time
import numpy as np
import mediapipe as mp
import pyautogui  # For controlling volume

wCam, hCam = 1280, 480
pTime = 0
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# Initialize MediaPipe Hand module
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    
    # Convert the image to RGB (required for MediaPipe)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # Get the landmarks
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                
                # Draw landmarks
                if id == 4:  # Thumb tip
                    thumb_x, thumb_y = cx, cy
                if id == 8:  # Index finger tip
                    index_x, index_y = cx, cy
                # You can visualize landmarks for debugging
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), -1)

            # Calculate the distance between the thumb and index finger
            distance = int(np.sqrt((thumb_x - index_x)**2 + (thumb_y - index_y)**2))
            
            # Map distance to volume control
            # You can adjust the range based on your preference
            if distance < 50:
                pyautogui.press('volumedown')  # Decrease volume
            elif distance > 150:
                pyautogui.press('volumeup')  # Increase volume

            # Show distance on the screen
            cv2.putText(img, f"Distance: {distance}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
            
            # Draw the hand landmarks on the image
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # Calculate FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    
    # Display the FPS on the screen
    cv2.putText(img, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    
    # Show the image
    cv2.imshow("Hand Volume Control", img)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
