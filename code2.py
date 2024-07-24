import cv2
import os
hog=cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
video_path=r"C:\Users\madhu\OneDrive\Desktop\DS internship\WhatsApp Video 2024-07-23 at 00.19.11_aa08a6e9.mp4"
cap=cv2.VideoCapture(video_path)
output_folder=r"C:\Users\madhu\OneDrive\Desktop\DS internship\Cropped objects"
os.makedirs(output_folder,exist_ok=True)
fps=cap.get(cv2.CAP_PROP_FPS)
total_frames=int(10*60*fps)
frame_id=0
while frame_id<total_frames:
    ret,frame=cap.read()
    if not ret:
        break
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    boxes, weights=hog.detectMultiScale(gray,winStride=(8,8))

    for i, (x,y,w,h) in enumerate(boxes):
        if weights[i] > 0.5:
            cropped_object=frame[y:y+h,x:x+w]
            object_filename=f'{output_folder}/frame_{frame_id}_object_{i}.jpg'
            cv2.imwrite(object_filename, cropped_object)

    frame_id+=1

cap.release()

os.listdir(output_folder)
