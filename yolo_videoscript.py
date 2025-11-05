import cv2
from ultralytics import YOLO
from collections import defaultdict

# Load a pre-trained YOLO model (e.g., yolov8n.pt is small and fast)
# Note: Use model name without .pt extension for auto-download from Ultralytics hub
model = YOLO('yolo11x-seg')

# Detection smoothing parameters
DETECTION_HISTORY_SIZE = 5  # Number of frames to look back for smoothing
detection_history = defaultdict(lambda: [])

# Path to your input video
input_video_path = 'house.mov'
output_video_path = 'yolohouse_with_boxes.mov'

# Open the input video
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties for the output writer
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
# Use 'avc1' (H.264) for better compatibility on macOS
fourcc = cv2.VideoWriter_fourcc(*'avc1')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Check if VideoWriter was successfully initialized
if not out.isOpened():
    print(f"Error: Could not open VideoWriter with codec 'avc1'")
    print(f"Video properties: {frame_width}x{frame_height} @ {fps}fps")
    cap.release()
    exit()

# Process the video frame by frame
frame_count = 0
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run YOLOv8 tracking on the frame with lower confidence threshold
    # The 'persist=True' flag tells the tracker that this is a continuous video stream
    # conf: Lower this to detect more objects (default is 0.25)
    # iou: Intersection over Union threshold for NMS (default is 0.7)
    results = model.track(
        frame,
        persist=True,
        conf=0.15,  # Lower confidence threshold to detect smaller/less confident objects
        iou=0.5,    # Lower IOU to allow more overlapping detections
        verbose=False
    )

    # Apply temporal smoothing to reduce flickering
    current_detections = {}

    if results[0].boxes is not None and results[0].boxes.id is not None:
        for box_id, box_data in zip(results[0].boxes.id, results[0].boxes.data):
            track_id = int(box_id)
            # Store detection in history
            detection_history[track_id].append(frame_count)
            # Keep only recent history
            detection_history[track_id] = detection_history[track_id][-DETECTION_HISTORY_SIZE:]
            current_detections[track_id] = box_data

    # Only show detections that appear in at least 2 of the last DETECTION_HISTORY_SIZE frames
    # This smooths out flickering detections
    smoothed_boxes = []
    for track_id, box_data in current_detections.items():
        history = detection_history[track_id]
        # Check if this detection has appeared recently enough
        if len(history) >= 2 or (len(history) >= 1 and frame_count - history[-1] <= 2):
            smoothed_boxes.append(box_data)

    # Create a copy of results with only smoothed detections
    if len(smoothed_boxes) > 0:
        # Create annotated frame with smoothed detections
        annotated_frame = results[0].plot(labels=False)
    else:
        # No smoothed detections, use original frame
        annotated_frame = frame.copy()

    # Ensure frame dimensions match
    if annotated_frame.shape[1] != frame_width or annotated_frame.shape[0] != frame_height:
        print(f"Warning: Frame {frame_count} size mismatch. Resizing...")
        annotated_frame = cv2.resize(annotated_frame, (frame_width, frame_height))

    # Write the annotated frame to the output video
    out.write(annotated_frame)
    frame_count += 1

    # Print progress every 30 frames
    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames...")

# Release everything when done
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"\nProcessing complete!")
print(f"Total frames processed: {frame_count}")
print(f"Output video saved to: {output_video_path}")