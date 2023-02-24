import cv2 
import numpy as np

import mediapipe as mp 

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

def _build_model():
    global mp_pose
    pose_estimator = mp_pose.Pose(
        static_image_mode=False, 
        model_complexity=1,
        enable_segmentation=False, 
        min_detection_confidence=0.5 
        )
    
    return pose_estimator

def _estimate_right_hand_angle(joints):
    
    wrist_x, wrist_y = joints.landmark[16].x, joints.landmark[16].y
    index_x, index_y = joints.landmark[20].x, joints.landmark[20].y
    
    estimation_score = (joints.landmark[16].visibility + joints.landmark[20].visibility) / 2
    hand_angle = -np.arctan2((index_y - wrist_y), (index_x - wrist_x)) * 180 / np.pi 
    
    return hand_angle, estimation_score

def _estimate_left_hand_angle(joints) :
    
    wrist_x, wrist_y = joints.landmark[15].x, joints.landmark[15].y
    index_x, index_y = joints.landmark[19].x, joints.landmark[19].y

    estimation_score = (joints.landmark[15].visibility + joints.landmark[19].visibility) / 2 
    hand_angle = -np.arctan2((wrist_y - index_y), (wrist_x - index_x)) * 180 / np.pi 

    return hand_angle, estimation_score

def _detect_hand_waving(joints):
    
    right_hand_angle, right_hand_estimation_score = _estimate_right_hand_angle(joints)
    left_hand_angle, left_hand_estimation_score = _estimate_left_hand_angle(joints)
    
    print(right_hand_estimation_score, left_hand_estimation_score)
    print(right_hand_angle, left_hand_angle)
    
    if right_hand_estimation_score > 0.5 :
        if 70 < right_hand_angle < 120 or -70< right_hand_angle < -60:
            print("right hand true")
            return True
    
    if left_hand_estimation_score > 0.5:
        if 60 < left_hand_angle < 70 or -120 < left_hand_angle < -70:
            print("left hand true")
            return True
    
    return False

def main():
    global mp_pose, mp_drawing, mp_drawing_styles
    
    predictor = _build_model()
    
    cap = cv2.VideoCapture(0)
    
    while True:
        
        _, frame = cap.read()
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = predictor.process(image)
        joints =results.pose_landmarks if results.pose_landmarks else None
        
        hand_angle = None
        
        if joints is not None:
            is_waving_hand = _detect_hand_waving(joints)
            
            if is_waving_hand:
                print("Hand is waving")
                frame = cv2.putText(frame,
                            text='Waving',
                            org=(50, 50),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1.0,
                            color=(0, 255, 0),
                            thickness=2,
                            lineType=cv2.LINE_4)
                
            mp_drawing.draw_landmarks(frame,
                                      joints, 
                                      mp_pose.POSE_CONNECTIONS, 
                                      landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                                      )
        
        cv2.imshow("Frame", frame)
        
        if cv2.waitKey(1) == 27:
            break
        
    predictor.close() 
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()
    
            
        
        
        