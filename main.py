import cv2
import os
import datetime

def detect_motion():
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = None
    recording = False
    
    _, frame1 = cap.read()
    _, frame2 = cap.read()
    
    save_path = "motion_captures"
    os.makedirs(save_path, exist_ok=True)
    
    motion_detected = False
    
    while True:
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_in_frame = any(cv2.contourArea(contour) > 1000 for contour in contours)
        
        if motion_in_frame and not motion_detected:
            print("Motion Started")
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            img_path = os.path.join(save_path, f"motion_start_{timestamp}.jpg")
            cv2.imwrite(img_path, frame1)
            
            video_path = os.path.join(save_path, f"motion_{timestamp}.avi")
            video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (frame1.shape[1], frame1.shape[0]))
            recording = True
            
            motion_detected = True
        
        if recording:
            video_writer.write(frame1)
        
        for contour in contours:
            if cv2.contourArea(contour) > 1000:
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        cv2.imshow("Motion Detection", frame1)
        frame1 = frame2
        _, frame2 = cap.read()
        
        if cv2.waitKey(1) & 0xFF == ord('\\'):
            print("Recording Stopped by User")
            break
    
    cap.release()
    if recording:
        video_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_motion()