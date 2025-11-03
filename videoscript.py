import cv2
import base64
import json
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
if not OPENROUTER_API_KEY:
    print("Error: OPENROUTER_API_KEY not found in .env file")
    print("Please add it to your .env file: OPENROUTER_API_KEY='your-api-key'")
    exit()

# Initialize OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

# Video paths
input_video_path = 'dirty.mov'
output_video_path = 'dirty_with_boxes.mov'

# Processing settings
PROCESS_EVERY_N_FRAMES = 1  # Process EVERY frame for maximum smoothness (~$1.80 per minute)

def build_detection_prompt(previous_detections=None):
    """Build prompt with optional context from previous frame"""
    if previous_detections and len(previous_detections) > 0:
        # Include previous detections for context
        prev_objects = [d.get("label", "unknown") for d in previous_detections]
        return f"""In the previous frame, you detected these objects: {", ".join(prev_objects)}.

Now analyze this NEW frame and report bounding boxes for ALL objects in the room you should track every single object, no matter how small it is.  (continue tracking the previous objects and add any new ones you see).

Use specific labels like: bed, dirty clothes, water bottle, nightstand, door, window, etc.

Return ONLY valid JSON in this exact format:
{{
  "bounding_boxes": [
    {{"label": "bed", "bbox_2d": [x1, y1, x2, y2]}},
    {{"label": "dirty clothes", "bbox_2d": [x1, y1, x2, y2]}}
  ]
}}"""
    else:
        # First frame - no context
        return """Report the bounding boxes of every object in the room with specific labels (e.g., bed, dirty clothes, water bottle, nightstand, door, window, etc.).

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

def get_bounding_boxes(frame, previous_detections=None):
    """Send frame to Qwen3-VL and get bounding box detections"""
    base64_image = encode_frame_to_base64(frame)
    response_text = None

    try:
        prompt = build_detection_prompt(previous_detections)

        response = client.chat.completions.create(
            model="qwen/qwen3-vl-32b-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.1,  # Lower temperature for more consistent outputs
        )

        # Extract the response text
        response_text = response.choices[0].message.content

        # Debug: print the raw response to see what we're getting
        print(f"Raw API response: {response_text[:200]}...")

        # Try to parse JSON from the response
        # The model might wrap it in markdown code blocks, so we'll handle that
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            json_str = response_text[json_start:json_end].strip()
        elif "```" in response_text:
            json_start = response_text.find("```") + 3
            json_end = response_text.find("```", json_start)
            json_str = response_text[json_start:json_end].strip()
        else:
            # Try to find JSON object or array directly
            if response_text.strip().startswith("["):
                # It's a JSON array
                json_str = response_text.strip()
            else:
                # Try to find JSON object
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                json_str = response_text[json_start:json_end]

        detections = json.loads(json_str)

        # Handle both formats: {"bounding_boxes": [...]} or directly [...]
        if isinstance(detections, dict):
            return detections.get("bounding_boxes", [])
        elif isinstance(detections, list):
            return detections
        else:
            return []

    except Exception as e:
        print(f"Error getting detections: {e}")
        if response_text:
            print(f"Response text: {response_text[:500]}")
        return []

def draw_bounding_boxes(frame, bounding_boxes):
    """Draw bounding boxes on frame"""
    frame_height, frame_width = frame.shape[:2]

    for detection in bounding_boxes:
        label = detection.get("label", "unknown")
        bbox = detection.get("bbox_2d", [])

        if len(bbox) == 4:
            # QWen3-VL returns coordinates in 0-1000 normalized space
            # Scale them to actual frame dimensions
            x1 = int((bbox[0] / 1000.0) * frame_width)
            y1 = int((bbox[1] / 1000.0) * frame_height)
            x2 = int((bbox[2] / 1000.0) * frame_width)
            y2 = int((bbox[3] / 1000.0) * frame_height)

            # Clamp coordinates to frame boundaries
            x1 = max(0, min(x1, frame_width - 1))
            y1 = max(0, min(y1, frame_height - 1))
            x2 = max(0, min(x2, frame_width - 1))
            y2 = max(0, min(y2, frame_height - 1))

            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1), (0, 255, 0), -1)

            # Draw label text
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return frame

def main():
    # Open the input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video properties: {frame_width}x{frame_height} @ {fps}fps")
    print(f"Total frames: {total_frames}")
    print(f"Processing every {PROCESS_EVERY_N_FRAMES} frames")

    # Create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    if not out.isOpened():
        print(f"Error: Could not open VideoWriter")
        cap.release()
        exit()

    # Process video
    frame_count = 0
    last_bounding_boxes = []  # Cache last detections

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Get detections for this frame or use cached ones
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            print(f"Processing frame {frame_count}/{total_frames}...")
            # Pass previous frame's detections for context
            bounding_boxes = get_bounding_boxes(frame, last_bounding_boxes if frame_count > 0 else None)
            last_bounding_boxes = bounding_boxes
            print(f"Found {len(bounding_boxes)} detections")
        else:
            bounding_boxes = last_bounding_boxes

        # Draw bounding boxes on frame
        annotated_frame = frame.copy()
        if bounding_boxes:
            annotated_frame = draw_bounding_boxes(annotated_frame, bounding_boxes)

        # Write frame to output
        out.write(annotated_frame)
        frame_count += 1

        # Progress update
        if frame_count % 30 == 0:
            print(f"Written {frame_count}/{total_frames} frames...")

    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"\nProcessing complete!")
    print(f"Total frames processed: {frame_count}")
    print(f"Output video saved to: {output_video_path}")

if __name__ == "__main__":
    main()
