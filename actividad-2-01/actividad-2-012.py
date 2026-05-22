import cv2
import pickle
import numpy as np

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Process Frame 10: Upscale by 1.5x
        if frame_count == 10:
            resized_10 = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
            cv2.imwrite('frame_10.png', resized_10)
            
        # Process Frame 30: Downscale by 0.5x
        elif frame_count == 30:
            resized_30 = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            cv2.imwrite('frame_30.png', resized_30)
        
        # Process Frame 50: Rotate 35 degrees around the center
        elif frame_count == 50:
            height, width = frame.shape[:2]
            cx, cy = width / 2.0, height / 2.0
            
            angle_rad = np.radians(35.0)
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            
            # Step 1: Move center to origin
            t_to_origin = np.eye(3, dtype=np.float64)
            t_to_origin[0, 2] = -cx
            t_to_origin[1, 2] = -cy
            
            # Step 2: Rotate
            rotation_matrix = np.eye(3, dtype=np.float64)
            rotation_matrix[0, 0] = cos_a
            rotation_matrix[0, 1] = sin_a
            rotation_matrix[1, 0] = -sin_a
            rotation_matrix[1, 1] = cos_a
            
            # Step 3: Move back to original center
            t_back_to_center = np.eye(3, dtype=np.float64)
            t_back_to_center[0, 2] = cx
            t_back_to_center[1, 2] = cy
            
            # Compose the full transformation matrix
            full_matrix = t_back_to_center @ rotation_matrix @ t_to_origin
            rotated_50 = cv2.warpAffine(frame, full_matrix[:2], (width, height))
            cv2.imwrite('frame_50.png', rotated_50)
            
            with open('rotation_matrix.pkl', 'wb') as f:
                pickle.dump(full_matrix, f)
            
        # Process Frame 70: Flip horizontally
        elif frame_count == 70:
            flipped_70 = cv2.flip(frame, 1)
            cv2.imwrite('frame_70.png', flipped_70)
            break
            
    cap.release()
    print("done again.")

video_file_path = './actividad-2-01/video.mp4'
process_video(video_file_path)