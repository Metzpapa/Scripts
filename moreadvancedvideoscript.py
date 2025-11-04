import cv2
import base64
import json
import os
import time
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
API_RATE_LIMIT_DELAY = 1.2  # Delay between API calls in seconds (50 requests/min to stay under 60/min limit)
MAX_API_RETRIES = 2  # Number of retries for failed API calls

def build_detection_prompt(previous_detections=None):
    """Build prompt with optional context from previous frame"""
    if previous_detections and len(previous_detections) > 0:
        # Include full previous detections with bounding boxes for better tracking
        prev_detections_json = json.dumps(previous_detections, indent=2)
        return f"""In the previous frame, you detected these objects with their bounding boxes:
{prev_detections_json}

Now analyze this NEW frame and update the bounding boxes for the objects you see. Only track distinct, separate objects - do not create multiple boxes for the same item.

IMPORTANT: Return a maximum of 15 unique objects. Do not duplicate boxes.

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
    start_time = time.time()

    # Time the encoding
    encode_start = time.time()
    base64_image = encode_frame_to_base64(frame)
    encode_time = time.time() - encode_start
    print(f"  [Timing] Frame encoding: {encode_time:.2f}s")

    response_text = None

    # Retry logic for API calls
    for attempt in range(MAX_API_RETRIES):
        try:
            if attempt > 0:
                print(f"  [Retry] Attempt {attempt + 1}/{MAX_API_RETRIES}")
                time.sleep(2)  # Wait 2 seconds before retrying

            # Time the prompt building
            prompt_start = time.time()
            prompt = build_detection_prompt(previous_detections)
            prompt_time = time.time() - prompt_start
            print(f"  [Timing] Prompt building: {prompt_time:.2f}s")
            print(f"  [Info] Prompt length: {len(prompt)} chars")
            print(f"\n{'='*80}")
            print(f"FULL PROMPT:")
            print(f"{'='*80}")
            print(prompt)
            print(f"{'='*80}\n")

            # Time the API call
            api_start = time.time()
            print(f"  [Status] Making API call...")

            try:
                stream = client.chat.completions.create(
                    model="qwen/qwen3-vl-235b-a22b-instruct",
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
                    max_tokens=4000,  # Limit output to prevent runaway generation
                    stream=True,  # Enable streaming
                    timeout=60.0,  # 60 second timeout
                )

                # Collect streamed response
                response_text = ""
                print(f"  [Status] Streaming response:")
                print(f"  {'='*80}")
                chunk_count = 0
                for chunk in stream:
                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, 'content') and delta.content:
                            response_text += delta.content
                            chunk_count += 1
                            # Print the actual content as it arrives
                            print(delta.content, end="", flush=True)

                print()  # New line after streaming
                print(f"  {'='*80}")
                api_time = time.time() - api_start
                print(f"  [Timing] API call: {api_time:.2f}s")
                print(f"  [Info] Received {chunk_count} chunks")
            except Exception as api_error:
                api_time = time.time() - api_start
                print(f"  [Timing] API call failed after: {api_time:.2f}s")
                print(f"  [Error] API error: {type(api_error).__name__}: {api_error}")
                raise  # Re-raise to be caught by outer exception handler

            print(f"  [Info] Response length: {len(response_text)} chars")

            # Debug: print the FULL response
            print(f"\n{'='*80}")
            print(f"FULL API RESPONSE:")
            print(f"{'='*80}")
            print(response_text)
            print(f"{'='*80}\n")

            # Time the parsing
            parse_start = time.time()
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
            parse_time = time.time() - parse_start
            print(f"  [Timing] JSON parsing: {parse_time:.2f}s")

            # Handle both formats: {"bounding_boxes": [...]} or directly [...]
            if isinstance(detections, dict):
                result = detections.get("bounding_boxes", [])
            elif isinstance(detections, list):
                result = detections
            else:
                result = []

            total_time = time.time() - start_time
            print(f"  [Timing] TOTAL frame processing: {total_time:.2f}s")

            # Rate limiting: wait before next API call
            print(f"  [Rate Limit] Waiting {API_RATE_LIMIT_DELAY}s before next request...")
            time.sleep(API_RATE_LIMIT_DELAY)

            return result

        except Exception as e:
            total_time = time.time() - start_time
            print(f"  [Timing] Attempt failed after: {total_time:.2f}s")
            print(f"  [Error] Error getting detections: {type(e).__name__}: {e}")
            if response_text:
                print(f"  Response text: {response_text[:500]}")

            # If this was the last attempt, return empty
            if attempt == MAX_API_RETRIES - 1:
                print(f"  [Error] All {MAX_API_RETRIES} attempts failed. Returning empty detections.")
                return []
            # Otherwise, continue to next retry attempt

    # Fallback (should never reach here, but just in case)
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
    successful_frames = 0
    failed_frames = 0

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Get detections for this frame or use cached ones
            if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                print(f"\nProcessing frame {frame_count}/{total_frames}...")
                # Pass previous frame's detections for context
                bounding_boxes = get_bounding_boxes(frame, last_bounding_boxes if frame_count > 0 else None)

                if bounding_boxes and len(bounding_boxes) > 0:
                    last_bounding_boxes = bounding_boxes
                    successful_frames += 1
                    print(f"✓ Found {len(bounding_boxes)} detections")
                else:
                    # Use previous frame's boxes if current frame failed
                    bounding_boxes = last_bounding_boxes
                    failed_frames += 1
                    print(f"✗ Failed to get detections, using previous frame's {len(bounding_boxes)} boxes")
            else:
                bounding_boxes = last_bounding_boxes

            # Draw bounding boxes on frame
            annotated_frame = frame.copy()
            if bounding_boxes:
                annotated_frame = draw_bounding_boxes(annotated_frame, bounding_boxes)

            # Write frame to output
            out.write(annotated_frame)
            frame_count += 1

            # Progress update and periodic flush
            if frame_count % 30 == 0:
                out.release()  # Release and reopen to flush to disk
                out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
                # Reopen in append mode by seeking to end
                print(f"📹 Written {frame_count}/{total_frames} frames (Success: {successful_frames}, Failed: {failed_frames})")
                print(f"   Video saved so far to: {output_video_path}")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user! Saving video with frames processed so far...")
    except Exception as e:
        print(f"\n\n⚠️  Error occurred: {e}")
        print("Saving video with frames processed so far...")
    finally:
        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        print(f"\n{'='*80}")
        print(f"Processing stopped!")
        print(f"Total frames processed: {frame_count}/{total_frames}")
        print(f"Successful API calls: {successful_frames}")
        print(f"Failed API calls: {failed_frames}")
        print(f"Output video saved to: {output_video_path}")
        print(f"{'='*80}")

if __name__ == "__main__":
    main()