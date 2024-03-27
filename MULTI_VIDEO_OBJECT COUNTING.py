import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
from threading import Thread

# Function to process each video and count vehicles
def process_video(video_path, index, frame_rate=6, video_speed=1):
    cap = cv2.VideoCapture(video_path)
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        count += 1
        if count % frame_rate != 0:
            continue

        frame = cv2.resize(frame, (640,360))

        try:
            # Detect objects in the frame with YOLO model
            bbox, label, conf = cv.detect_common_objects(frame, confidence=0.5, model='yolov4-tiny')

            # Filter out labels other than 'car' and 'truck'
            relevant_labels = ['car', 'truck']
            relevant_indices = [i for i, lab in enumerate(label) if lab in relevant_labels]
            bbox = [bbox[i] for i in relevant_indices]
            label = [label[i] for i in relevant_indices]
            conf = [conf[i] for i in relevant_indices]

            # Count vehicles
            car_count = label.count('car')
            truck_count = label.count('truck')

            total_count = car_count+truck_count

            # Display output with counts
            output_image = draw_bbox(frame, bbox, label, conf)
            cv2.putText(output_image, f"Car Count: {car_count}", (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 2)
            cv2.putText(output_image, f"Truck Count: {truck_count}", (10, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 2)
            cv2.putText(output_image, f"Total Count: {total_count}", (10, 90), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 2)
            cv2.imshow(f"Video {index+1}", output_image)

        except Exception as e:
            print(f"Error processing frame: {e}")
            cv2.imshow(f"Video {index+1}", frame)

        if cv2.waitKey(int(1000 / (frame_rate * video_speed))) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# Paths to the four videos
video_paths = [
    "C:\\Users\\91801\\OneDrive\\Desktop\\EXSEL PROJECT\\TRAFFIC\\VIDEOS\\Traffic video 2.mp4",
    "C:\\Users\\91801\\OneDrive\\Desktop\\EXSEL PROJECT\\TRAFFIC\\VIDEOS\\Traffic video1.mp4",
    "C:\\Users\\91801\\OneDrive\\Desktop\\EXSEL PROJECT\\TRAFFIC\\VIDEOS\\Traffic video 3.mp4",
    "C:\\Users\\91801\\OneDrive\\Desktop\\EXSEL PROJECT\\TRAFFIC\\VIDEOS\\Traffic Video 4.mp4"
]

# Start a thread for each video
threads = []
for index, video_path in enumerate(video_paths):
    thread = Thread(target=process_video, args=(video_path, index,6))  # Change frame_rate and video_speed as needed
    threads.append(thread)
    thread.start()

# Wait for all threads to finish
for thread in threads:
    thread.join()
