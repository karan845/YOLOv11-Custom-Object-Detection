import torch
import cv2
import numpy as np
import yt_dlp as youtube_dl
import argparse
from ultralytics import YOLO
import os

# Load YOLO model
MODEL_PATH = "yolov11_custom.pt"
model = YOLO(MODEL_PATH)


def detect_objects(frame):
    """
    Runs object detection on a given frame using YOLO.
    
    Args:
        frame: Image frame (numpy array)
    
    Returns:
        Processed frame with bounding boxes
    """
    results = model(frame)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  
            conf = float(box.conf[0])  
            label = result.names[int(box.cls[0])]  

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame


def process_image(image_path):
    """
    Reads an image, performs object detection, displays and saves the result.
    """
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read image.")
        return

    processed_image = detect_objects(image)
    cv2.imshow("Detected Objects", processed_image)

    # Save the output image
    output_path = f"output_{os.path.basename(image_path)}"
    cv2.imwrite(output_path, processed_image)
    print(f"Processed image saved as {output_path}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_webcam():
    """
    Captures webcam feed, performs real-time object detection, and saves the output video.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    output_video_path = "output_webcam.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        processed_frame = detect_objects(frame)
        cv2.imshow("Webcam Detection", processed_frame)
        out.write(processed_frame)  # Save the frame to video

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Webcam video saved as {output_video_path}")


def get_youtube_video_url(youtube_url):
    """
    Extracts the best video stream URL using yt_dlp.
    """
    ydl_opts = {'format': 'best', 'quiet': True}
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info['url']


def process_youtube(youtube_url):
    """
    Streams a YouTube video, runs object detection, and saves the output video.
    """
    try:
        video_url = get_youtube_video_url(youtube_url)
        cap = cv2.VideoCapture(video_url)

        if not cap.isOpened():
            print("Error: Could not open YouTube stream.")
            return

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        output_video_path = "output_youtube.avi"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (frame_width, frame_height))

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to retrieve video frame.")
                break

            processed_frame = detect_objects(frame)
            cv2.imshow("YouTube Video Detection", processed_frame)
            out.write(processed_frame)  # Save the frame to video

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"YouTube video saved as {output_video_path}")

    except Exception as e:
        print(f"Error processing YouTube video: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object Detection using YOLOv11")
    parser.add_argument("--image", type=str, help="Path to image file")
    parser.add_argument("--youtube", type=str, help="YouTube video URL")
    parser.add_argument("--webcam", action="store_true", help="Use webcam for detection")

    args = parser.parse_args()

    if args.image:
        process_image(args.image)
    elif args.youtube:
        process_youtube(args.youtube)
    elif args.webcam:
        process_webcam()
    else:
        print("Please provide an input source: --image, --youtube, or --webcam.")
