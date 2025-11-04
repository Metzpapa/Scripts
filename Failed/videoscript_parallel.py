import cv2
import base64
import json
import os
import asyncio
import argparse
import time
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

# Networking libs for OpenRouter key introspection
import aiohttp

# OpenAI SDK (pointed at OpenRouter)
from openai import AsyncOpenAI

# =========================
# Load environment variables
# =========================
load_dotenv()

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
if not OPENROUTER_API_KEY:
    print("Error: OPENROUTER_API_KEY not found in .env file")
    print("Please add it to your .env file: OPENROUTER_API_KEY='your-api-key'")
    exit(1)

# =========================
# OpenRouter client
# =========================
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

client = AsyncOpenAI(
    base_url=f"{OPENROUTER_BASE}",
    api_key=OPENROUTER_API_KEY
)

# Target model (you can change this)
TARGET_MODEL = "qwen/qwen3-vl-32b-instruct"   # e.g., add ':free' if you are using a free variant
MODEL_IS_FREE = TARGET_MODEL.endswith(":free")

# =========================
# Video paths
# =========================
input_video_path = 'HOUSE.mov'
output_video_path = 'HOUSE_with_boxes.mov'
data_file_path = 'HOUSE_detections.json'
error_log_path = 'HOUSE_errors.json'

# =========================
# Processing settings
# =========================
PROCESS_EVERY_N_FRAMES = 1        # Process EVERY frame
MAX_CONCURRENT_REQUESTS = 10      # Upper bound for the adaptive limiter
SAVE_CHECKPOINT_EVERY = 50        # Save checkpoint every N frames

# =========================
# Adaptive Rate Limiter
# =========================

class RateLimitSignal(Exception):
    """Raised when we detect a rate limit (429 or in-stream error)."""
    def __init__(self, retry_after: float | None = None, msg: str = ""):
        super().__init__(msg)
        self.retry_after = retry_after

async def fetch_openrouter_key_info(api_key: str) -> dict:
    """
    GET /key to learn live credit/limit info.
    This is the official way to inspect your key and plan behavior dynamically.
    """
    headers = {"Authorization": f"Bearer {api_key}"}
    async with aiohttp.ClientSession() as sess:
        async with sess.get(f"{OPENROUTER_BASE}/key", headers=headers) as r:
            r.raise_for_status()
            return await r.json()

def is_stream_rate_limit_error(resp_obj: dict | None) -> bool:
    """
    Detect OpenRouter 'rate-limit as a 200 response' pattern where error appears inside a normal-looking body.
    Works with objects surfaced by the OpenAI SDK (model_dump/dict) as well.
    """
    if not isinstance(resp_obj, dict):
        return False

    if resp_obj.get("error"):
        msg = (resp_obj["error"].get("message") or "").lower()
        return "rate limit" in msg

    choices = resp_obj.get("choices") or []
    if choices and (choices[0].get("finish_reason") == "error"):
        msg = (choices[0].get("message") or {}).get("content", "")
        return "rate limit" in str(msg).lower()

    return False

class AdaptiveLimiter:
    """
    A feedback-driven concurrency controller with optional RPM token bucket.

    - Uses an asyncio.Condition to gate the number of in-flight tasks to cur_c.
    - If using a :free model, it additionally enforces a strict 20 RPM token bucket.
    - On success: gentle additive increase, up to max_c.
    - On rate limit: multiplicative decrease and respect Retry-After header if provided.
    """
    def __init__(
        self,
        start_concurrency: int = 4,
        min_concurrency: int = 1,
        max_concurrency: int = 64,
        enforce_free_20rpm: bool = False
    ):
        self.min_c = max(1, int(min_concurrency))
        self.max_c = max(self.min_c, int(max_concurrency))
        self.cur_c = max(self.min_c, min(int(start_concurrency), self.max_c))

        self.add_step = 1          # additive increase per healthy cycle
        self.mult_decrease = 0.5   # multiplicative drop on RL

        # Concurrency gating
        self._cond = asyncio.Condition()
        self._active = 0  # number of active (in-flight) tasks

        # RPM token bucket for :free models (20 requests per minute)
        self.enforce_free_20rpm = enforce_free_20rpm
        self.bucket_capacity = 20 if enforce_free_20rpm else None
        self.bucket_tokens = 20 if enforce_free_20rpm else None
        self.bucket_refill_interval = 60.0
        self.bucket_last_refill = time.monotonic()

    def _refill_bucket_if_needed(self):
        if not self.enforce_free_20rpm:
            return
        now = time.monotonic()
        elapsed = now - self.bucket_last_refill
        if elapsed >= self.bucket_refill_interval:
            self.bucket_tokens = self.bucket_capacity
            self.bucket_last_refill = now

    async def _acquire_bucket(self):
        if not self.enforce_free_20rpm:
            return
        while True:
            self._refill_bucket_if_needed()
            if self.bucket_tokens and self.bucket_tokens > 0:
                self.bucket_tokens -= 1
                return
            # wait until the next refill
            to_next = self.bucket_refill_interval - (time.monotonic() - self.bucket_last_refill)
            await asyncio.sleep(max(0.05, to_next))

    async def acquire(self):
        """
        Wait until both:
          - we have under 'cur_c' in-flight tasks, and
          - (if :free) a token is available in the RPM bucket.
        """
        await self._acquire_bucket()

        start_ts = time.monotonic()
        async with self._cond:
            while self._active >= self.cur_c:
                await self._cond.wait()
            self._active += 1
            return start_ts

    def release(self, start_ts: float | None = None):
        """
        Mark a task complete and notify others waiting.
        """
        async def _release():
            async with self._cond:
                if self._active > 0:
                    self._active -= 1
                self._cond.notify_all()
        # schedule release but don't await here to avoid blocking caller
        asyncio.get_running_loop().create_task(_release())

    async def on_success(self):
        """
        On healthy completion: gently ramp concurrency up to max_c.
        """
        async with self._cond:
            new_c = min(self.max_c, self.cur_c + self.add_step)
            if new_c != self.cur_c:
                self.cur_c = new_c
                self._cond.notify_all()

    async def on_rate_limit(self, retry_after: float | None):
        """
        On RL signal: cut concurrency immediately, and sleep if server told us how long.
        """
        async with self._cond:
            decreased = max(self.min_c, int(max(1, self.cur_c) * self.mult_decrease))
            if decreased < self.cur_c:
                self.cur_c = decreased
            self._cond.notify_all()

        if retry_after and retry_after > 0:
            await asyncio.sleep(retry_after)
        else:
            # conservative default pause based on current concurrency
            await asyncio.sleep(1.0 + 0.1 * self.cur_c)

limiter: AdaptiveLimiter | None = None

async def init_limiter():
    """
    Initialize the limiter with live account info and model context.
    """
    try:
        key_info = await fetch_openrouter_key_info(OPENROUTER_API_KEY)
        data = key_info.get("data", {})
        remaining = data.get("limit_remaining")
        daily = data.get("usage_daily")
        print(f"[OpenRouter] Remaining credits: {remaining}, daily usage: {daily}")
    except Exception as e:
        print(f"[OpenRouter] Warning: could not read /key info ({e}). Proceeding with defaults.")

    start = min(MAX_CONCURRENT_REQUESTS, 4)  # cautious start; will ramp up
    global limiter
    limiter = AdaptiveLimiter(
        start_concurrency=start,
        min_concurrency=1,
        max_concurrency=MAX_CONCURRENT_REQUESTS,
        enforce_free_20rpm=MODEL_IS_FREE
    )
    print(f"[Limiter] start={start}, max={MAX_CONCURRENT_REQUESTS}, :free_enforced={MODEL_IS_FREE}")

# =========================
# Prompt / utils
# =========================

def build_detection_prompt():
    """Build prompt for detection - simplified for parallel processing"""
    return """Report the bounding boxes of every single object in the room with specific labels (e.g., small_plant, wilted_plant, painting, lamp, etc.).

IMPORTANT: Look carefully at the TOP of the image. If you see any drywall damage on the ceiling (especially when there are two chairs and a tree visible), mark it with label "ceiling_damage".

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

def log_failed_response(frame_number, response_text, error_message, json_attempt, api_attempt, final=False):
    """Log failed API responses for debugging"""
    try:
        # Load existing error log
        if os.path.exists(error_log_path):
            with open(error_log_path, 'r') as f:
                error_log = json.load(f)
        else:
            error_log = {"errors": []}

        # Create error entry
        error_entry = {
            "frame_number": frame_number,
            "json_attempt": json_attempt,
            "api_attempt": api_attempt,
            "error_message": error_message,
            "response_text": response_text[:1000] if response_text else None,  # Limit to 1000 chars
            "final_failure": final
        }

        error_log["errors"].append(error_entry)

        # Save error log
        with open(error_log_path, 'w') as f:
            json.dump(error_log, f, indent=2)

    except Exception as e:
        print(f"  Warning: Could not log error for frame {frame_number}: {e}")

# =========================
# Core API call
# =========================

async def get_bounding_boxes(frame_data, frame_number, retry_count=3, json_retry_count=5):
    """Send frame to model and get bounding box detections (async)

    Args:
        frame_data: Tuple of (base64_image, frame_number)
        frame_number: Frame number to process
        retry_count: Number of retries for API errors
        json_retry_count: Number of retries specifically for JSON parsing errors
    """
    base64_image, frame_number = frame_data
    response_text = None
    last_error = None

    for json_attempt in range(json_retry_count):
        for api_attempt in range(retry_count):
            start_ts = None
            try:
                # Acquire concurrency slot (+ RPM token if :free)
                start_ts = await limiter.acquire()

                prompt = build_detection_prompt()

                response = await client.chat.completions.create(
                    model=TARGET_MODEL,
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
                )

                # Detect "rate limit" that arrives as a successful 200 with error payload
                raw = getattr(response, "model_dump", None)
                resp_obj = raw() if callable(raw) else getattr(response, "dict", lambda: {})()
                if is_stream_rate_limit_error(resp_obj):
                    raise RateLimitSignal(msg="In-stream rate limit")

                # Extract the response text
                response_text = response.choices[0].message.content

                # Try to parse JSON from the response
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
                        json_str = response_text.strip()
                    else:
                        json_start = response_text.find("{")
                        json_end = response_text.rfind("}") + 1
                        json_str = response_text[json_start:json_end]

                detections = json.loads(json_str)

                # Handle both formats
                limiter.release(start_ts)
                await limiter.on_success()

                if isinstance(detections, dict):
                    return frame_number, detections.get("bounding_boxes", [])
                elif isinstance(detections, list):
                    return frame_number, detections
                else:
                    return frame_number, []

            except RateLimitSignal as rl:
                limiter.release(start_ts)
                await limiter.on_rate_limit(rl.retry_after)
                # retry same attempt
                continue

            except Exception as e:
                # If transport exposes status/headers, honor Retry-After on 429
                retry_after = None
                msg = str(e).lower()
                # Heuristic detection for RL; if you can access e.status/e.headers, improve here
                if "429" in msg or "rate limit" in msg:
                    limiter.release(start_ts)
                    await limiter.on_rate_limit(retry_after)
                    continue

                last_error = e
                limiter.release(start_ts)
                if api_attempt < retry_count - 1:
                    # transient: back off then retry
                    await asyncio.sleep(2 ** api_attempt)
                    continue
                else:
                    # Exhausted API retries for this JSON attempt
                    if json_attempt < json_retry_count - 1:
                        await asyncio.sleep(1.5 ** json_attempt)
                    break  # go to next JSON attempt

        # JSON attempt exhausted; log and try new request cycle
        print(f"\n  [JSON Parse Error] Frame {frame_number}, JSON attempt {json_attempt + 1}/{json_retry_count}")
        log_failed_response(frame_number, response_text, "JSON parse failure", json_attempt, api_attempt)

    # All retries exhausted
    print(f"\n  [FAILED] Frame {frame_number} - All {json_retry_count * retry_count} attempts exhausted")
    log_failed_response(frame_number, response_text, str(last_error), json_retry_count, retry_count, final=True)
    return frame_number, []

# =========================
# Drawing / IO helpers
# =========================

def draw_bounding_boxes(frame, bounding_boxes, color=(0, 255, 0), thickness=2):
    """Draw bounding boxes on frame with customizable styling"""
    frame_height, frame_width = frame.shape[:2]

    for detection in bounding_boxes:
        label = detection.get("label", "unknown")
        bbox = detection.get("bbox_2d", [])

        if len(bbox) == 4:
            # Qwen3-VL returns coordinates in 0-1000 normalized space
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
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # Draw label background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1), color, -1)

            # Draw label text
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return frame

def load_detection_data(data_file):
    """Load existing detection data from JSON file"""
    if os.path.exists(data_file):
        try:
            with open(data_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse {data_file}, starting fresh")
            return {"metadata": {}, "frames": {}}
    return {"metadata": {}, "frames": {}}

def save_detection_data(data_file, detection_data):
    """Save detection data to JSON file"""
    with open(data_file, 'w') as f:
        json.dump(detection_data, f, indent=2)

def get_last_processed_frame(detection_data):
    """Get the last frame number that was processed"""
    if detection_data.get("frames"):
        return max(int(frame_num) for frame_num in detection_data["frames"].keys())
    return -1

# =========================
# Parallel processing
# =========================

async def process_video_parallel(force_reprocess=False):
    """Process video in parallel and save detection data"""
    # Load existing detection data (for resuming)
    if force_reprocess:
        detection_data = {"metadata": {}, "frames": {}}
        last_processed = -1
        print("Force reprocessing enabled - starting fresh")
    else:
        detection_data = load_detection_data(data_file_path)
        last_processed = get_last_processed_frame(detection_data)

    # Open the input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit(1)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Save metadata
    detection_data["metadata"] = {
        "input_video": input_video_path,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "fps": fps,
        "total_frames": total_frames,
        "process_every_n_frames": PROCESS_EVERY_N_FRAMES
    }

    print(f"Video properties: {frame_width}x{frame_height} @ {fps}fps")
    print(f"Total frames: {total_frames}")
    print(f"Processing every {PROCESS_EVERY_N_FRAMES} frames")
    print(f"Concurrency upper bound: {MAX_CONCURRENT_REQUESTS}")
    if last_processed >= 0:
        print(f"Resuming from frame {last_processed + 1}")

    # Extract all frames first
    print("\nExtracting frames...")
    frames_to_process = []
    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            # Skip if already processed
            if frame_count <= last_processed:
                frame_count += 1
                continue

            base64_image = encode_frame_to_base64(frame)
            frames_to_process.append((base64_image, frame_count))

        frame_count += 1

    cap.release()

    if not frames_to_process:
        print("No frames to process!")
        return

    print(f"Extracted {len(frames_to_process)} frames to process")
    print(f"\nInitializing adaptive limiter...")
    await init_limiter()

    print(f"Processing frames in parallel...")

    results = []
    with tqdm(total=len(frames_to_process), desc="Processing frames", unit="frame") as pbar:
        # To control memory, process in rolling batches; batch size scales with max concurrency
        batch_size = max(1, MAX_CONCURRENT_REQUESTS * 5)

        for i in range(0, len(frames_to_process), batch_size):
            batch = frames_to_process[i:i + batch_size]
            tasks = []

            for frame_data, frame_num in batch:
                async def process_single(fd=frame_data, fn=frame_num):
                    result = await get_bounding_boxes((fd, fn), fn)
                    pbar.update(1)
                    return result

                tasks.append(process_single())

            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

            # Periodic checkpoint
            if (i // batch_size) % 5 == 0:
                for frame_num, detections in batch_results:
                    detection_data["frames"][str(frame_num)] = detections
                save_detection_data(data_file_path, detection_data)
                print(f"\n  [Checkpoint saved at batch {i // batch_size}]")

    # Save all results
    for frame_num, detections in results:
        detection_data["frames"][str(frame_num)] = detections

    save_detection_data(data_file_path, detection_data)

    # Count failures
    failed_frames = [frame_num for frame_num, detections in results if not detections]
    success_count = len(results) - len(failed_frames)

    print(f"\n=== Processing complete! ===")
    print(f"Total frames processed: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(failed_frames)}")
    if failed_frames:
        print(f"Failed frames: {failed_frames[:10]}{'...' if len(failed_frames) > 10 else ''}")
        print(f"Check error log for details: {error_log_path}")
    print(f"Detection data saved to: {data_file_path}")
    print(f"\nTo rebuild video with this data, run:")
    print(f"  python videoscript_parallel.py --rebuild")

# =========================
# Rebuild video
# =========================

def rebuild_video(color=(0, 255, 0), thickness=2):
    """Rebuild video from saved detection data"""
    # Load detection data
    if not os.path.exists(data_file_path):
        print(f"Error: Detection data file not found: {data_file_path}")
        print("Run processing first to generate detection data.")
        exit(1)

    detection_data = load_detection_data(data_file_path)
    metadata = detection_data.get("metadata", {})
    frames_data = detection_data.get("frames", {})

    if not metadata:
        print("Error: No metadata found in detection data")
        exit(1)

    # Open the input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit(1)

    # Get video properties from metadata
    frame_width = metadata["frame_width"]
    frame_height = metadata["frame_height"]
    fps = metadata["fps"]
    total_frames = metadata["total_frames"]

    print(f"Rebuilding video from saved data...")
    print(f"Video properties: {frame_width}x{frame_height} @ {fps}fps")
    print(f"Total frames in data: {len(frames_data)}")

    # Create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    if not out.isOpened():
        print(f"Error: Could not open VideoWriter")
        cap.release()
        exit(1)

    # Rebuild video
    frame_count = 0
    last_bounding_boxes = []

    with tqdm(total=total_frames, desc="Rebuilding video", unit="frame") as pbar:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Get detections from saved data
            if str(frame_count) in frames_data:
                bounding_boxes = frames_data[str(frame_count)]
                last_bounding_boxes = bounding_boxes
            else:
                # Use last known detections if current frame not in data
                bounding_boxes = last_bounding_boxes

            # Draw bounding boxes on frame
            annotated_frame = frame.copy()
            if bounding_boxes:
                annotated_frame = draw_bounding_boxes(annotated_frame, bounding_boxes, color=color, thickness=thickness)

            # Write frame to output
            out.write(annotated_frame)
            frame_count += 1
            pbar.update(1)

    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"\n=== Rebuild complete! ===")
    print(f"Total frames: {frame_count}")
    print(f"Output video saved to: {output_video_path}")

# =========================
# Main
# =========================

def main():
    global MAX_CONCURRENT_REQUESTS, MODEL_IS_FREE, TARGET_MODEL

    parser = argparse.ArgumentParser(description='Process video with parallel object detection (adaptive limiter)')
    parser.add_argument('--rebuild', action='store_true',
                       help='Rebuild video from saved detection data')
    parser.add_argument('--force', action='store_true',
                       help='Force reprocess all frames (ignore existing detection data)')
    parser.add_argument('--color', type=str, default='0,255,0',
                       help='Bounding box color in BGR format (default: 0,255,0 for green)')
    parser.add_argument('--thickness', type=int, default=2,
                       help='Bounding box line thickness (default: 2)')
    parser.add_argument('--concurrency', type=int, default=MAX_CONCURRENT_REQUESTS,
                       help=f'Upper bound on concurrent requests (default: {MAX_CONCURRENT_REQUESTS})')
    parser.add_argument('--model', type=str, default=TARGET_MODEL,
                       help=f'Model ID (default: {TARGET_MODEL}). Append :free to enforce 20 RPM token bucket.')

    args = parser.parse_args()

    if args.rebuild:
        # Parse color
        color_parts = args.color.split(',')
        if len(color_parts) == 3:
            color = tuple(int(c) for c in color_parts)
        else:
            print("Warning: Invalid color format, using default green")
            color = (0, 255, 0)

        rebuild_video(color=color, thickness=args.thickness)
    else:
        # Update settings
        MAX_CONCURRENT_REQUESTS = max(1, int(args.concurrency))
        TARGET_MODEL = args.model
        MODEL_IS_FREE = TARGET_MODEL.endswith(":free")

        # Run async processing
        asyncio.run(process_video_parallel(force_reprocess=args.force))

if __name__ == "__main__":
    main()
