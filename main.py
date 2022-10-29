from turtle import right
import cv2
import mediapipe as mp
import numpy as np
import json


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

exercise = input("qual ex deseja fazer?")

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

cap = cv2.VideoCapture(0)

# Curl counter variables
counter = 0 
stage = None
numb = 0

## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        angles = []
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            # Calculate angle
            
            left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            #angles.append(left_elbow_angle)

            right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            #angles.append(right_elbow_angle)

            right_shoulder_angle = calculate_angle(right_hip, right_shoulder, right_elbow)
            angles.append(right_shoulder_angle)

            center = [int((left_shoulder[0]+right_shoulder[0])/2), int((left_shoulder[1]+right_shoulder[1])/2)]
            backbone_angle = calculate_angle(center, right_hip, right_ankle)
            angles.append(backbone_angle)

            left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
            #angles.append(left_knee_angle)
            
            right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
            angles.append(right_knee_angle)
            
            left_shoulder_angle = calculate_angle(left_hip, left_shoulder, left_elbow)
            #angles.append(left_shoulder_angle)
            
            right_betweenlegs_angle = calculate_angle(right_knee, right_hip, left_knee)
            angles.append(right_betweenlegs_angle)

            
            # Visualize angle
            cv2.putText(image, str(int(left_elbow_angle)), 
                           tuple(np.multiply(left_elbow, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA
                                )
            cv2.putText(image, str(int(right_elbow_angle)), 
                           tuple(np.multiply(right_elbow, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA
                                )
            cv2.putText(image, str(int(right_knee_angle)), 
                           tuple(np.multiply(right_knee, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA
                                )
            cv2.putText(image, str(int(left_knee_angle)), 
                           tuple(np.multiply(left_knee, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA
                                )
            cv2.putText(image, str(int(right_shoulder_angle)), 
                           tuple(np.multiply(right_shoulder, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA
                                )                           
            cv2.putText(image, str(int(left_shoulder_angle)), 
                           tuple(np.multiply(left_shoulder, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA
                                )
            
            # Curl counter logic
            if exercise == 'biceps':
                if left_elbow_angle > 100 and right_elbow_angle > 100:
                    stage = "down"
                if left_elbow_angle < 40 and right_elbow_angle < 40 and stage =='down':
                    stage="up"
                    counter +=1

            elif exercise == 'agachamento':
                if left_knee_angle > 100 and right_knee_angle > 100 and stage == 'down':
                    stage = 'up'
                    counter += 1
                if left_knee_angle < 90 and right_knee_angle < 90:
                    stage = 'down'

            elif exercise == 'desenvolvimento':
                if left_shoulder_angle > 120 and right_shoulder_angle > 120 and stage == 'down':
                    stage = 'up'
                    counter += 1
                if left_shoulder_angle < 120:
                    stage = 'down'
                       
        except:
            pass
            
        # Rep data
        cv2.putText(image, 'COUNT', (15,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), (10,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        # Stage data
        cv2.putText(image, 'STAGE', (80,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, stage, 
                    (60,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               
        
        cv2.imshow('Mediapipe Feed', image)

        with open('data/keypoints'+str(numb)+'.json', 'w') as f:
            if angles != []:
                json.dump(angles, f)
                numb += 1
            else:
                pass

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()