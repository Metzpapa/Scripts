import cv2
import base64
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
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
input_video_path = 'house.mov'
output_video_path = 'house2_with_boxes.mov'
detections_json_path = 'house2_detections.json'  # Save detections for later reprocessing

# Processing settings
PROCESS_EVERY_N_FRAMES = 3  # Process every 3 frames and interpolate between them
MAX_CONCURRENT_REQUESTS = 5  # Start with 5 concurrent requests
MAX_RETRIES = 3  # Maximum retries for failed JSON parsing

# Special frame range for additional prompt
SPECIAL_FRAME_START = 210
SPECIAL_FRAME_END = 308
SPECIAL_PROMPT_ADDITION = " There is damage on the drywall on the ceiling in some of the frames. Please check and mark of its there. "

# Rate limiting state
class RateLimiter:
    def __init__(self, initial_concurrency=5):
        self.lock = Lock()
        self.concurrency = initial_concurrency
        self.last_rate_limit_time = 0
        self.request_times = []

    def on_rate_limit(self):
        """Called when we hit a rate limit - reduce concurrency"""
        with self.lock:
            self.concurrency = max(1, int(self.concurrency * 0.5))
            self.last_rate_limit_time = time.time()
            print(f"Rate limit hit! Reducing concurrency to {self.concurrency}")

    def on_success(self):
        """Called on successful request - slowly increase concurrency"""
        with self.lock:
            # Only increase if we haven't been rate limited recently (last 30 seconds)
            if time.time() - self.last_rate_limit_time > 30:
                self.concurrency = min(20, self.concurrency + 1)

    def get_concurrency(self):
        """Get current concurrency limit"""
        with self.lock:
            return self.concurrency

    def should_wait(self):
        """Check if we need to back off after rate limit"""
        with self.lock:
            if time.time() - self.last_rate_limit_time < 2:
                return True
            return False

rate_limiter = RateLimiter(MAX_CONCURRENT_REQUESTS)

def build_detection_prompt(previous_detections=None, frame_idx=None):
    """Build prompt with optional context from previous frame"""
    # Check if this frame is in the special range
    special_addition = ""
    if frame_idx is not None and SPECIAL_FRAME_START <= frame_idx <= SPECIAL_FRAME_END:
        special_addition = SPECIAL_PROMPT_ADDITION

    if previous_detections and len(previous_detections) > 0:
        # Include previous detections for context
        prev_objects = [d.get("label", "unknown") for d in previous_detections]
        return f"""In the previous frame, you detected these objects: {", ".join(prev_objects)}.

Now analyze this NEW frame and report bounding boxes for ALL objects in the room you should track every single object, no matter how small it is.  (continue tracking the previous objects and add any new ones you see).

Use specific labels like: bed, dirty clothes, water bottle, nightstand, door, window, etc.{special_addition}

Return ONLY valid JSON in this exact format:
{{
  "bounding_boxes": [
    {{"label": "bed", "bbox_2d": [x1, y1, x2, y2]}},
    {{"label": "dirty clothes", "bbox_2d": [x1, y1, x2, y2]}}
  ]
}}"""
    else:
        # First frame - no context
        return f"""Report the bounding boxes of every object in the room with specific labels (e.g., bed, dirty clothes, water bottle, nightstand, door, window, etc.).{special_addition}

Return ONLY valid JSON in this exact format:
{{
  "bounding_boxes": [
    {{"label": "bed", "bbox_2d": [x1, y1, x2, y2]}},
    {{"label": "dirty clothes", "bbox_2d": [x1, y1, x2, y2]}}
  ]
}}"""

def encode_frame_to_base64(frame):
    """Convert OpenCV frame to base64 encoded string"""
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

def repair_json_response(json_str):
    """Attempt to repair common JSON formatting issues from the API"""
    import re

    # Fix 1: {"label": "window", "130, 240...} -> {"label": "window", "bbox_2d": [130, 240...
    # This handles the quoted coordinates issue
    pattern1 = r'(\{"label":\s*"[^"]+",\s*)"(\d+,\s*\d+)'
    repaired = re.sub(pattern1, r'\1"bbox_2d": [\2', json_str)

    # Fix 2: {"label": "window", [130, 240...} -> {"label": "window", "bbox_2d": [130, 240...
    # This handles missing "bbox_2d": with bracket already present
    pattern2 = r'(\{"label":\s*"[^"]+",\s*)(\[\d+)'
    repaired = re.sub(pattern2, r'\1"bbox_2d": \2', repaired)

    # Fix 3: {"label": "window", 130, 240...} -> {"label": "window", "bbox_2d": [130, 240...
    # This handles missing both "bbox_2d": and bracket
    pattern3 = r'(\{"label":\s*"[^"]+",\s*)(?!"bbox_2d":)(\d+,\s*\d+)'
    repaired = re.sub(pattern3, r'\1"bbox_2d": [\2', repaired)

    return repaired

def get_bounding_boxes(frame, previous_detections=None, retry_count=0, frame_idx=None):
    """Send frame to Qwen3-VL and get bounding box detections with retry logic"""
    base64_image = encode_frame_to_base64(frame)
    response_text = None

    try:
        # Check if we should wait due to recent rate limiting
        if rate_limiter.should_wait():
            time.sleep(2)

        prompt = build_detection_prompt(previous_detections, frame_idx)

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

        # Check for error in response
        if hasattr(response.choices[0], 'finish_reason') and response.choices[0].finish_reason == "error":
            raise Exception("API returned error finish_reason")

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

        # Attempt to repair common JSON issues before parsing
        json_str = repair_json_response(json_str)

        detections = json.loads(json_str)

        # Handle both formats: {"bounding_boxes": [...]} or directly [...]
        if isinstance(detections, dict):
            result = detections.get("bounding_boxes", [])
        elif isinstance(detections, list):
            result = detections
        else:
            result = []

        # Check if we got zero detections and retry
        if len(result) == 0 and retry_count < MAX_RETRIES:
            print(f"Zero objects detected on attempt {retry_count + 1}/{MAX_RETRIES}. Retrying...")
            time.sleep(1)
            return get_bounding_boxes(frame, previous_detections, retry_count + 1, frame_idx)

        # Signal success to rate limiter
        rate_limiter.on_success()
        return result

    except json.JSONDecodeError as e:
        print(f"JSON parsing error on attempt {retry_count + 1}/{MAX_RETRIES}: {e}")
        if response_text:
            print(f"Invalid JSON response: {response_text[:500]}")

        # Retry if we haven't exceeded max retries
        if retry_count < MAX_RETRIES:
            print(f"Retrying frame (attempt {retry_count + 2}/{MAX_RETRIES})...")
            time.sleep(1)  # Brief pause before retry
            return get_bounding_boxes(frame, previous_detections, retry_count + 1, frame_idx)
        else:
            print(f"Max retries ({MAX_RETRIES}) reached. Returning empty detections.")
            return []

    except Exception as e:
        error_msg = str(e)
        print(f"Error getting detections: {error_msg}")
        if response_text:
            print(f"Response text: {response_text[:500]}")

        # Check if it's a rate limit error (429 or error in response)
        if "429" in error_msg or "rate" in error_msg.lower() or "error" in error_msg.lower():
            rate_limiter.on_rate_limit()
            # Wait and retry
            if retry_count < MAX_RETRIES:
                time.sleep(3)  # Longer pause for rate limits
                return get_bounding_boxes(frame, previous_detections, retry_count + 1, frame_idx)

        # For other errors, retry if we haven't exceeded max retries
        if retry_count < MAX_RETRIES:
            print(f"Retrying frame (attempt {retry_count + 2}/{MAX_RETRIES})...")
            time.sleep(1)
            return get_bounding_boxes(frame, previous_detections, retry_count + 1, frame_idx)

        return []

def draw_bounding_boxes(frame, bounding_boxes):
    """Draw bounding boxes on frame with color coding"""
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

            # Choose color based on label content
            # Red for anything with "damage" in the label, light blue for everything else
            if "damage" in label.lower():
                color = (0, 0, 255)  # Red in BGR
            else:
                color = (255, 200, 100)  # Light blue in BGR

            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1), color, -1)

            # Draw label text
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return frame

def interpolate_detections(prev_detections, next_detections, alpha):
    """Interpolate bounding boxes between two frames"""
    if not prev_detections or not next_detections:
        # If either is empty, just return the non-empty one or empty list
        return next_detections if next_detections else prev_detections

    # Simple interpolation: blend the bounding boxes
    interpolated = []

    # Try to match objects by label
    for prev_det in prev_detections:
        prev_label = prev_det.get("label", "")
        prev_bbox = prev_det.get("bbox_2d", [])

        # Find matching object in next frame
        matching_next = None
        for next_det in next_detections:
            if next_det.get("label", "") == prev_label:
                matching_next = next_det
                break

        if matching_next and len(prev_bbox) == 4:
            next_bbox = matching_next.get("bbox_2d", [])
            if len(next_bbox) == 4:
                # Interpolate coordinates
                interp_bbox = [
                    int(prev_bbox[0] * (1 - alpha) + next_bbox[0] * alpha),
                    int(prev_bbox[1] * (1 - alpha) + next_bbox[1] * alpha),
                    int(prev_bbox[2] * (1 - alpha) + next_bbox[2] * alpha),
                    int(prev_bbox[3] * (1 - alpha) + next_bbox[3] * alpha)
                ]
                interpolated.append({"label": prev_label, "bbox_2d": interp_bbox})

    # Add any new objects from next frame that weren't in prev
    for next_det in next_detections:
        next_label = next_det.get("label", "")
        if not any(d.get("label") == next_label for d in prev_detections):
            interpolated.append(next_det)

    return interpolated

def process_frame_task(frame_data):
    """Process a single frame (for concurrent execution)"""
    frame_idx, frame, previous_detections = frame_data
    bounding_boxes = get_bounding_boxes(frame, previous_detections, frame_idx=frame_idx)
    return frame_idx, bounding_boxes

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
    print(f"Processing every {PROCESS_EVERY_N_FRAMES} frames with concurrent requests")

    # First pass: Read all frames and store them
    print("Reading all frames into memory...")
    frames = []
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frames.append(frame)

    cap.release()
    print(f"Loaded {len(frames)} frames")

    # Second pass: Process frames concurrently
    print("Processing frames with concurrent API requests...")
    frame_detections = {}  # Store detections by frame index
    frames_to_process = []

    # Prepare frames to process
    for i in range(0, len(frames), PROCESS_EVERY_N_FRAMES):
        # Get previous frame's detections for context (if available)
        prev_detections = frame_detections.get(i - PROCESS_EVERY_N_FRAMES, []) if i > 0 else None
        frames_to_process.append((i, frames[i], prev_detections))

    # Process frames in batches with dynamic concurrency
    processed_count = 0
    batch_start = 0

    while batch_start < len(frames_to_process):
        # Get current concurrency limit
        current_concurrency = rate_limiter.get_concurrency()
        batch_end = min(batch_start + current_concurrency, len(frames_to_process))
        batch = frames_to_process[batch_start:batch_end]

        print(f"\nProcessing batch {batch_start}-{batch_end-1} with concurrency {current_concurrency}")

        # Process batch concurrently
        with ThreadPoolExecutor(max_workers=current_concurrency) as executor:
            # Submit all tasks in the batch
            future_to_frame = {executor.submit(process_frame_task, frame_data): frame_data[0]
                             for frame_data in batch}

            # Collect results as they complete
            for future in as_completed(future_to_frame):
                frame_idx = future_to_frame[future]
                try:
                    idx, bounding_boxes = future.result()
                    frame_detections[idx] = bounding_boxes
                    processed_count += 1
                    print(f"Frame {idx}/{total_frames} complete - Found {len(bounding_boxes)} objects ({processed_count}/{len(frames_to_process)} frames done)")
                except Exception as e:
                    print(f"Error processing frame {frame_idx}: {e}")
                    frame_detections[frame_idx] = []

        batch_start = batch_end

    # Save detections to JSON file for later reprocessing
    print("\nSaving detection data to JSON...")
    detection_data = {
        "video_info": {
            "input_path": input_video_path,
            "width": frame_width,
            "height": frame_height,
            "fps": fps,
            "total_frames": len(frames)
        },
        "detections": {
            str(frame_idx): detections
            for frame_idx, detections in frame_detections.items()
        }
    }

    with open(detections_json_path, 'w') as f:
        json.dump(detection_data, f, indent=2)
    print(f"Detection data saved to: {detections_json_path}")

    # Third pass: Create output video with annotations
    print("\nCreating output video with bounding boxes...")
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    if not out.isOpened():
        print(f"Error: Could not open VideoWriter")
        exit()

    last_bounding_boxes = []
    for frame_idx, frame in enumerate(frames):
        # Get detections for this frame
        if frame_idx in frame_detections:
            # Direct detection available
            bounding_boxes = frame_detections[frame_idx]
            last_bounding_boxes = bounding_boxes
        else:
            # Interpolate between nearest processed frames
            # Find previous and next processed frames
            prev_idx = None
            next_idx = None

            for i in range(frame_idx - 1, -1, -1):
                if i in frame_detections:
                    prev_idx = i
                    break

            for i in range(frame_idx + 1, len(frames)):
                if i in frame_detections:
                    next_idx = i
                    break

            if prev_idx is not None and next_idx is not None:
                # Interpolate between prev and next
                prev_detections = frame_detections[prev_idx]
                next_detections = frame_detections[next_idx]
                # Calculate alpha (0 = prev, 1 = next)
                alpha = (frame_idx - prev_idx) / (next_idx - prev_idx)
                bounding_boxes = interpolate_detections(prev_detections, next_detections, alpha)
            elif prev_idx is not None:
                # Use previous frame
                bounding_boxes = frame_detections[prev_idx]
            elif next_idx is not None:
                # Use next frame
                bounding_boxes = frame_detections[next_idx]
            else:
                # No detections available
                bounding_boxes = last_bounding_boxes

        # Draw bounding boxes on frame
        annotated_frame = frame.copy()
        if bounding_boxes:
            annotated_frame = draw_bounding_boxes(annotated_frame, bounding_boxes)

        # Write frame to output
        out.write(annotated_frame)

        # Progress update
        if (frame_idx + 1) % 30 == 0:
            print(f"Written {frame_idx + 1}/{total_frames} frames...")

    # Cleanup
    out.release()
    cv2.destroyAllWindows()

    print(f"\nProcessing complete!")
    print(f"Total frames processed: {len(frames)}")
    print(f"Output video saved to: {output_video_path}")

if __name__ == "__main__":
    main()
