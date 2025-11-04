import cv2
import base64
import json
import os
import argparse
import numpy as np
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import concurrent.futures
from tqdm import tqdm
import supervision as sv

# --- Configuration ---

# Load environment variables from .env file
load_dotenv()

# Initialize OpenRouter client
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
if not OPENROUTER_API_KEY:
    print("Error: OPENROUTER_API_KEY not found in .env file")
    exit()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

# Video paths
input_video_path = 'HOUSE.mov'
output_video_path = 'HOUSE_with_tracking.mov'
data_file_path = 'HOUSE_detections.json'

# --- New Processing Settings ---
KEYFRAME_INTERVAL = 5       # Process one frame every N frames with the VLM
MAX_CONCURRENT_REQUESTS = 10 # Max parallel requests to OpenRouter
SAVE_CHECKPOINT_EVERY = 10   # Save checkpoint every N processed keyframes

# --- VLM Interaction Functions ---

def build_detection_prompt(previous_detections=None):
    """Build prompt with optional context from previous frame"""
    if previous_detections and len(previous_detections) > 0:
        prev_objects = [d.get("label", "unknown") for d in previous_detections]
        return f"""In the previous image, you detected these objects: {", ".join(prev_objects)}.

Now analyze this new image and report bounding boxes for all objects in the room. You should track every object that can be seen in the image.
Use specific labels like: small_plant, wilted_plant, painting, lamp, ect.

It may not be in this frame, but in one of the images I give you, there's going to be damage in the top part of the screen that is drywall damage on the ceiling. When you see that mark it, but it's not in every image, so it might not be in this image. You should see it when you see two chairs and a tree, and on the top of the screen, there'll be damage.

Return ONLY valid JSON in this exact format:
{{
  "bounding_boxes": [
    {{"label": "bed", "bbox_2d": [x1, y1, x2, y2]}},
    {{"label": "dirty clothes", "bbox_2d": [x1, y1, x2, y2]}}
  ]
}}"""
    else:
        return """Report the bounding boxes of every single object in the room with specific labels (e.g., small_plant, wilted_plant, painting, lamp, ect.).

Return ONLY valid JSON in this exact format:
{
  "bounding_boxes": [
    {"label": "bed", "bbox_2d": [x1, y1, x2, y2]},
    {"label": "dirty clothes", "bbox_2d": [x1, y1, x2, y2]}
  ]
}"""

def encode_frame_to_base64(frame):
    """Convert OpenCV frame to base64 encoded string"""
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

def get_bounding_boxes(frame, frame_number, previous_detections=None):
    """
    Send frame to Qwen3-VL and get bounding box detections.
    Now returns frame_number along with detections for parallel processing.
    """
    base64_image = encode_frame_to_base64(frame)
    response_text = None

    try:
        prompt = build_detection_prompt(previous_detections)
        response = client.chat.completions.create(
            model="qwen/qwen3-vl-32b-instruct",
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ],
            temperature=0.1,
        )
        response_text = response.choices[0].message.content

        if "```json" in response_text:
            json_str = response_text.split("```json\n")[1].split("\n```")[0]
        elif "```" in response_text:
            json_str = response_text.split("```\n")[1].split("\n```")[0]
        else:
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            json_str = response_text[json_start:json_end]

        detections = json.loads(json_str)
        bounding_boxes = detections.get("bounding_boxes", []) if isinstance(detections, dict) else detections
        return frame_number, bounding_boxes

    except Exception as e:
        print(f"Error processing frame {frame_number}: {e}")
        if response_text:
            print(f"Response text for frame {frame_number}: {response_text[:500]}")
        return frame_number, []

# --- Data Handling Functions ---

def load_detection_data(data_file):
    if os.path.exists(data_file):
        try:
            with open(data_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {"metadata": {}, "frames": {}}
    return {"metadata": {}, "frames": {}}

def save_detection_data(data_file, detection_data):
    with open(data_file, 'w') as f:
        json.dump(detection_data, f, indent=2)

def get_last_processed_frame(detection_data):
    if detection_data.get("frames"):
        return max(int(frame_num) for frame_num in detection_data["frames"].keys())
    return -1

# --- Main Processing Logic ---

def process_video_keyframes():
    """
    Processes only keyframes from the video in parallel batches.
    """
    detection_data = load_detection_data(data_file_path)
    
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    detection_data["metadata"] = {
        "input_video": input_video_path, "frame_width": frame_width,
        "frame_height": frame_height, "fps": fps, "total_frames": total_frames,
        "keyframe_interval": KEYFRAME_INTERVAL
    }

    # Identify keyframes that still need processing
    existing_frames = {int(k) for k in detection_data["frames"].keys()}
    keyframes_to_process = [
        i for i in range(0, total_frames, KEYFRAME_INTERVAL) if i not in existing_frames
    ]

    if not keyframes_to_process:
        print("All keyframes already processed. To re-process, delete the JSON file.")
        print("Now run with --rebuild to generate the tracked video.")
        cap.release()
        return

    print(f"Found {len(keyframes_to_process)} keyframes to process.")
    
    processed_count = 0
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
            future_to_frame = {}
            
            for frame_num in keyframes_to_process:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                success, frame = cap.read()
                if success:
                    future = executor.submit(get_bounding_boxes, frame, frame_num, None)
                    future_to_frame[future] = frame_num

            # Process results as they complete
            progress_bar = tqdm(concurrent.futures.as_completed(future_to_frame), total=len(future_to_frame), desc="Processing Keyframes")
            for future in progress_bar:
                frame_number, bounding_boxes = future.result()
                if bounding_boxes:
                    detection_data["frames"][str(frame_number)] = bounding_boxes
                    print(f"Frame {frame_number}: Found {len(bounding_boxes)} detections.")
                
                processed_count += 1
                if processed_count % SAVE_CHECKPOINT_EVERY == 0:
                    save_detection_data(data_file_path, detection_data)
                    print(f"  [Checkpoint saved. {processed_count} keyframes processed.]")

    except KeyboardInterrupt:
        print("\nInterrupted. Saving progress...")
    finally:
        save_detection_data(data_file_path, detection_data)
        cap.release()
        print("\n=== Keyframe processing complete! ===")
        print(f"Detection data saved to: {data_file_path}")
        print(f"\nTo track and rebuild the video, run:")
        print(f"  python {Path(__file__).name} --rebuild")


def rebuild_video_with_tracking(color=(0, 255, 0), thickness=2):
    """
    Rebuilds video using ByteTrack to interpolate between keyframes.
    """
    if not os.path.exists(data_file_path):
        print(f"Error: Detection data file not found: {data_file_path}")
        exit()

    detection_data = load_detection_data(data_file_path)
    metadata = detection_data.get("metadata", {})
    frames_data = detection_data.get("frames", {})

    if not metadata or not frames_data:
        print("Error: No metadata or frame data found.")
        exit()

    cap = cv2.VideoCapture(input_video_path)
    frame_width, frame_height, fps, total_frames = metadata["frame_width"], metadata["frame_height"], metadata["fps"], metadata["total_frames"]

    fourcc = cv2.VideoWriter_fourcc(*'avc1') # Use 'mp4v' if 'avc1' fails
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Initialize ByteTrack
    tracker = sv.ByteTrack()
    
    # supervision.Color expects RGB, but our input `color` is BGR (OpenCV standard).
    b, g, r = color
    sv_color = sv.Color(r=r, g=g, b=b)
    
    # Initialize Supervision Annotators for drawing
    box_annotator = sv.BoxAnnotator(color=sv_color, thickness=thickness)
    label_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK, text_scale=0.5, text_thickness=1)

    print("Rebuilding video with tracking...")
    progress_bar = tqdm(range(total_frames), desc="Tracking & Rebuilding")

    for frame_count in progress_bar:
        success, frame = cap.read()
        if not success:
            break

        detections_sv = sv.Detections.empty()
        # If the current frame is a keyframe with detections, use them
        if str(frame_count) in frames_data:
            keyframe_detections = frames_data[str(frame_count)]
            
            bboxes = []
            labels = []
            for det in keyframe_detections:
                bbox = det.get("bbox_2d")
                if len(bbox) == 4:
                    x1 = int((bbox[0] / 1000.0) * frame_width)
                    y1 = int((bbox[1] / 1000.0) * frame_height)
                    x2 = int((bbox[2] / 1000.0) * frame_width)
                    y2 = int((bbox[3] / 1000.0) * frame_height)
                    bboxes.append([x1, y1, x2, y2])
                    labels.append(det.get("label", "unknown"))
            
            if bboxes:
                # --- CORRECTED LINE ---
                # Use the Detections constructor with the xyxy keyword argument
                detections_sv = sv.Detections(xyxy=np.array(bboxes))
                # --- END CORRECTION ---
                detections_sv.data['label'] = labels

        # Update the tracker with detections (or an empty object for interpolation)
        tracked_detections = tracker.update_with_detections(detections_sv)

        # Create labels for tracked objects, including their ID
        tracked_labels = [
            f"#{tracker_id} {det.get('label', '')}"
            for tracker_id, det in zip(tracked_detections.tracker_id, tracked_detections.data.values())
        ]

        # Annotate the frame with tracked boxes and labels
        annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=tracked_detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=tracked_detections, labels=tracked_labels)

        out.write(annotated_frame)

    cap.release()
    out.release()
    print(f"\n=== Rebuild complete! ===")
    print(f"Output video with tracking saved to: {output_video_path}")


def main():
    parser = argparse.ArgumentParser(description='Process video with VLM on keyframes and track objects, or rebuild from saved data.')
    parser.add_argument('--rebuild', action='store_true', help='Rebuild video using tracking from saved keyframe detection data.')
    parser.add_argument('--color', type=str, default='0,255,0', help='Bounding box color in BGR format (e.g., "0,255,0" for green).')
    parser.add_argument('--thickness', type=int, default=2, help='Bounding box line thickness.')
    args = parser.parse_args()

    if args.rebuild:
        try:
            color_parts = [int(c) for c in args.color.split(',')]
            if len(color_parts) != 3:
                raise ValueError
            color_tuple = tuple(color_parts)
        except (ValueError, TypeError):
            print(f"Warning: Invalid color format '{args.color}'. Using default green.")
            color_tuple = (0, 255, 0)
            
        rebuild_video_with_tracking(color=color_tuple, thickness=args.thickness)
    else:
        process_video_keyframes()

if __name__ == "__main__":
    main()