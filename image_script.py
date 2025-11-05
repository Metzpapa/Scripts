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

# Image paths - can be a single path or list of paths
IMAGE_PATHS = [
    'tepper.png',
    # Add more image paths as needed
]

# Output directory for processed images
OUTPUT_DIR = 'output_images'

def build_detection_prompt():
    """Build prompt for object detection with color coding"""
    return """Report the bounding boxes of every object in the image with specific labels. Try to detect every single object, including vents, etc. Also, there's a missing door handle on one of the booths it is the one with the hole in the door. You should mark that there's a missing door handle. Mark every single thing you see, including stains. This should include any damages on the floor, any damage you see anywhere. You dont have to label the entire floor and dont label the stuff in the super far background that is hard to recognize. Label all booths as simple "Booth". report a maxmimum of 20 bounding boxes. do not duplicate bounding boxes
    

For each object, include a "color" field with one of these values:
- "green" - for objects in good/excellent condition, clean, new, or well-maintained
- "orange" - for objects in acceptable/neutral condition, normal wear, or regular objects
- "red" - for objects that are damaged, broken, dirty, stained, or have problems

Return ONLY valid JSON in this exact format:
{
  "bounding_boxes": [
    {"label": "bed", "bbox_2d": [x1, y1, x2, y2], "color": "orange"},
    {"label": "damaged wall", "bbox_2d": [x1, y1, x2, y2], "color": "red"}
  ]
}"""

def encode_image_to_base64(image_path):
    """Convert image file to base64 encoded string"""
    # Read the image
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Could not read image: {image_path}")

    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8'), frame

def get_bounding_boxes(image_path):
    """Send image to Qwen3-VL and get bounding box detections with streaming"""
    print(f"Processing image: {image_path}")
    print("Encoding image to base64...")

    base64_image, frame = encode_image_to_base64(image_path)
    print(f"Image encoded. Size: {len(base64_image)} bytes")

    try:
        prompt = build_detection_prompt()
        print("Sending request to API with streaming enabled...")
        print("Streaming response:\n")

        import time
        start_time = time.time()

        # Enable streaming
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
            temperature=0.1,
            timeout=60,  # 1 minute timeout
            stream=True,  # Enable streaming
        )

        # Collect the full response from streaming chunks
        response_text = ""
        print("-" * 80)

        for chunk in stream:
            # Check for mid-stream errors
            if hasattr(chunk, 'error') and chunk.error:
                error_msg = chunk.error.get('message', 'Unknown error')
                print(f"\n\nStream error: {error_msg}")
                if chunk.choices and chunk.choices[0].finish_reason == 'error':
                    print("Stream terminated due to error")
                return [], frame

            # Extract content from chunk
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    content = delta.content
                    response_text += content
                    # Print the token as it arrives
                    print(content, end="", flush=True)

        print("\n" + "-" * 80)

        elapsed_time = time.time() - start_time
        print(f"\nStreaming complete in {elapsed_time:.2f} seconds")
        print(f"Total response length: {len(response_text)} characters\n")

        print("Parsing JSON response...")
        # Parse JSON from the response
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            json_str = response_text[json_start:json_end].strip()
        elif "```" in response_text:
            json_start = response_text.find("```") + 3
            json_end = response_text.find("```", json_start)
            json_str = response_text[json_start:json_end].strip()
        else:
            # Try to find JSON object directly
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            json_str = response_text[json_start:json_end]

        detections = json.loads(json_str)

        # Handle both formats: {"bounding_boxes": [...]} or directly [...]
        if isinstance(detections, dict):
            result = detections.get("bounding_boxes", [])
        elif isinstance(detections, list):
            result = detections
        else:
            result = []

        print(f"Found {len(result)} objects in image")
        return result, frame

    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Invalid JSON response: {response_text[:500]}")
        return [], frame
    except Exception as e:
        print(f"Error getting detections: {e}")
        return [], frame

def get_color_from_label(color_label):
    """
    Convert color label from model to BGR color tuple
    """
    color_map = {
        "green": (0, 255, 0),      # Green - good condition
        "orange": (0, 165, 255),   # Orange - neutral/acceptable
        "red": (0, 0, 255)         # Red - damaged/problematic
    }

    # Default to orange if color not recognized
    return color_map.get(color_label.lower(), (0, 165, 255))

def draw_bounding_boxes(frame, bounding_boxes):
    """Draw bounding boxes on frame with sentiment-based color coding"""
    frame_height, frame_width = frame.shape[:2]

    # Calculate dynamic sizing based on image dimensions
    # Box thickness scales with image size
    box_thickness = max(3, int(min(frame_width, frame_height) / 400))

    # Font scale also scales with image size
    font_scale = max(0.8, min(frame_width, frame_height) / 1500)
    font_thickness = max(2, int(font_scale * 2))

    for detection in bounding_boxes:
        label = detection.get("label", "unknown")
        bbox = detection.get("bbox_2d", [])
        color_label = detection.get("color", "orange")  # Get color from model

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

            # Get color from model's color label
            color = get_color_from_label(color_label)

            # Draw rectangle with thicker lines
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)

            # Draw label background with larger font
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            label_padding = int(font_scale * 10)

            # Make sure label background is visible
            label_y_top = max(label_size[1] + label_padding, y1 - label_size[1] - label_padding)
            cv2.rectangle(frame, (x1, label_y_top),
                         (x1 + label_size[0] + label_padding, y1), color, -1)

            # Draw label text with white color for better visibility
            cv2.putText(frame, label, (x1 + int(label_padding/2), y1 - int(label_padding/2)),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

    return frame

def process_image(image_path):
    """Process a single image and save the result"""
    # Get bounding boxes
    bounding_boxes, frame = get_bounding_boxes(image_path)

    # Draw bounding boxes
    if bounding_boxes:
        annotated_frame = draw_bounding_boxes(frame.copy(), bounding_boxes)
    else:
        annotated_frame = frame

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate output path
    basename = os.path.basename(image_path)
    name, ext = os.path.splitext(basename)
    output_path = os.path.join(OUTPUT_DIR, f"{name}_with_boxes{ext}")

    # Save annotated image
    print(f"Saving annotated image to: {output_path}")
    success = cv2.imwrite(output_path, annotated_frame)
    if not success:
        raise Exception(f"Failed to write image to {output_path}")
    print(f"Successfully saved annotated image\n")

    return output_path, bounding_boxes

def main():
    print(f"Processing {len(IMAGE_PATHS)} image(s)...\n")

    results = []
    for image_path in IMAGE_PATHS:
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue

        try:
            output_path, bounding_boxes = process_image(image_path)
            results.append({
                "input_path": image_path,
                "output_path": output_path,
                "detections": bounding_boxes
            })
        except Exception as e:
            print(f"Error processing {image_path}: {e}\n")

    # Save all results to JSON
    results_json_path = os.path.join(OUTPUT_DIR, "detections.json")
    print(f"\nSaving results to JSON: {results_json_path}")
    try:
        with open(results_json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Successfully saved results to JSON")
    except Exception as e:
        print(f"Error saving JSON: {e}")

    print(f"\nProcessing complete!")
    print(f"Processed {len(results)} image(s)")
    print(f"Results saved to: {results_json_path}")

if __name__ == "__main__":
    main()
