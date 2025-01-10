import cv2
import mediapipe as mp
import time

wCam, hCam = 1280, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id,lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                if id == 4:
                    thumb_x, thumb_y = cx, cy
                if id == 8:
                    index_x, index_y = cx, cy
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), -1)
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # Display FPS on the image
    cv2.putText(img, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # Show the image with landmarks and FPS
    cv2.imshow("Hand Volume Control", img)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
