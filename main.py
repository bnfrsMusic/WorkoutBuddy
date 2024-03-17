import cv2
import mediapipe as mp
import numpy as np


def calculate_angle(a, b, c):
    """Calculates the angle between three points."""
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

landmark_hips= [
    (11, 23, 25),  # Left side
    (12, 24, 26)   # Right side
]

landmark_arms= [
    (12, 14, 16),  # Left side
    (11, 13, 15)   # Right side
]

numPushups = 0
armAffirmation = False
hipAffirmation = False


while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    frame.flags.writeable = False
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame)

    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    if results.pose_landmarks:
        #print(numPushups)
        
        for hips in landmark_hips:
            points = [results.pose_landmarks.landmark[i] for i in hips]
            coords = [(int(point.x * frame.shape[1]), int(point.y * frame.shape[0])) for point in points]

            angle = calculate_angle(coords[0], coords[1], coords[2])


            color = (0, 0, 255)  # RED for non-straight
            if 160 <= angle <= 200:  # Adjust as necessary for "almost straight"
                color = (0, 255, 0)  # Green for almost straight or straight
                hipAffirmation = True
            else:
                hipAffirmation = False
            
            cv2.line(frame, coords[0], coords[1], color, 3)
            cv2.line(frame, coords[1], coords[2], color, 3)
            for coord in coords:
                cv2.circle(frame, coord, 4, color, -1)
        
        for arms in landmark_arms:
            points = [results.pose_landmarks.landmark[i] for i in arms]
            coords = [(int(point.x * frame.shape[1]), int(point.y * frame.shape[0])) for point in points]
            angle = calculate_angle(coords[0], coords[1], coords[2])
            color = (0, 255, 255)  # Green for non-straight
            if 80 <= angle <= 110 and (coords[1][0] <= (coords[2][0] + 100) and coords[1][0] >= (coords[2][0] - 100)):  # Adjust as necessary for "almost 90"
                color = (0, 255, 0)  # blue for almost straight or straight
                armAffirmation = True
            else:
                armAffirmation = False

            cv2.line(frame, coords[0], coords[1], color, 3)
            cv2.line(frame, coords[1], coords[2], color, 3)
            for coord in coords:
                cv2.circle(frame, coord, 4, color, -1)
        
        if armAffirmation == True and hipAffirmation == True:
            numPushups += 1

        
    cv2.imshow('MediaPipe Pose', frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
