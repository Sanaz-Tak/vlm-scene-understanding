import torch
import subprocess
import time
import json
import cv2
import numpy as np
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig


# ── Config ────────────────────────────────────────────────────────────────────
DEVICE_NAME      = "Logi C615 HD WebCam"
CLIP_DURATION    = 3        # seconds per clip
MAX_NEW_TOKENS   = 200
LOOP_COUNT       = None     # set to an int to limit iterations, or None to run forever
DARK_THRESHOLD   = 20       # mean pixel brightness (0-255) below which clip is skipped
MAX_PARSE_RETRY  = 1        # number of retries with simplified prompt on JSON parse failure
# ─────────────────────────────────────────────────────────────────────────────

PROMPT = """Analyze this video clip and respond ONLY with a JSON object in this exact format:
{
  "danger_detected": true or false,
  "description": "one sentence describing what you see",
  "confidence": "high" or "medium" or "low"
}
Do not include any other text."""

SIMPLE_PROMPT = 'Reply with ONLY this JSON, no other text: {"danger_detected": false, "description": "YOUR_DESCRIPTION_HERE", "confidence": "low"}'


def capture_clip(device_name: str, output_path: str, seconds: int) -> float:
    """Capture a clip and return capture duration in seconds."""
    t0 = time.perf_counter()
    cmd = [
        "ffmpeg.exe", "-y",
        "-f", "dshow", "-i", f"video={device_name}",
        "-t", str(seconds),
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")
    return elapsed


def is_dark(clip_path: str, threshold: int = DARK_THRESHOLD) -> bool:
    """Return True if the clip's mean brightness is below the threshold."""
    cap = cv2.VideoCapture(clip_path)
    brightnesses = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightnesses.append(gray.mean())
    cap.release()
    if not brightnesses:
        return True
    mean_brightness = np.mean(brightnesses)
    return mean_brightness < threshold


def parse_output(raw: str) -> dict | None:
    """Extract JSON from model output. Returns None on failure."""
    text = raw.split("assistant")[-1].strip()
    try:
        start = text.index("{")
        end   = text.rindex("}") + 1
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        return None


def load_model(model_name: str, device: str):
    print(f"Loading {model_name} on {device} (4-bit) ...")
    t0 = time.perf_counter()
    quant_config = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name, quantization_config=quant_config, device_map=device
    )
    processor = AutoProcessor.from_pretrained(model_name)
    print(f"Model loaded in {time.perf_counter() - t0:.1f}s\n")
    return model, processor


def infer_with_prompt(model, processor, device: str, clip_path: str, prompt: str) -> tuple[str, float]:
    """Run inference with a given prompt. Returns (raw output, inference time)."""
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": clip_path},
            {"type": "text",  "text": prompt},
        ],
    }]
    prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt_text, videos=[clip_path], return_tensors="pt").to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    inference_time = time.perf_counter() - t0

    raw = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return raw, inference_time


def infer(model, processor, device: str, clip_path: str) -> tuple[dict, float, bool]:
    """
    Run inference with automatic retry on JSON parse failure.
    Returns (result dict, total inference time, retried flag).
    """
    raw, inf_time = infer_with_prompt(model, processor, device, clip_path, PROMPT)
    result = parse_output(raw)

    if result is not None:
        return result, inf_time, False

    # Retry with simplified prompt
    for _ in range(MAX_PARSE_RETRY):
        raw2, inf_time2 = infer_with_prompt(model, processor, device, clip_path, SIMPLE_PROMPT)
        inf_time += inf_time2
        result = parse_output(raw2)
        if result is not None:
            return result, inf_time, True

    # All retries failed — return safe fallback
    return {"danger_detected": None, "description": "parse failed", "confidence": None}, inf_time, True


def run_pipeline(model, processor, device: str):
    clip_path       = "clip.mp4"
    iteration       = 0
    skipped         = 0
    retried         = 0
    parse_failures  = 0
    total_capture   = 0.0
    total_inference = 0.0

    print("Starting pipeline. Press Ctrl+C to stop.\n")
    print(f"{'#':<5} {'Capture(s)':<12} {'Infer(s)':<12} {'Danger':<8} {'Conf':<8} Description")
    print("-" * 90)

    try:
        while LOOP_COUNT is None or iteration < LOOP_COUNT:
            iteration += 1

            # Capture
            try:
                cap_time = capture_clip(DEVICE_NAME, clip_path, CLIP_DURATION)
            except RuntimeError as e:
                print(f"[{iteration}] Capture error: {e}")
                continue

            total_capture += cap_time

            # Black frame check
            if is_dark(clip_path):
                skipped += 1
                print(f"{iteration:<5} {cap_time:<12.2f} {'—':<12} {'DARK':<8} {'—':<8} skipped (dark frame)")
                continue

            # Infer
            result, inf_time, did_retry = infer(model, processor, device, clip_path)
            total_inference += inf_time
            if did_retry:
                retried += 1

            danger = result.get("danger_detected")
            conf   = result.get("confidence", "")
            desc   = result.get("description", "")

            if danger is None:
                parse_failures += 1

            flag   = "⚠ YES" if danger else ("no" if danger is False else "?")
            suffix = " [retry]" if did_retry else ""
            print(f"{iteration:<5} {cap_time:<12.2f} {inf_time:<12.2f} {flag:<8} {str(conf):<8} {desc}{suffix}")

    except KeyboardInterrupt:
        print("\n\nStopped by user.")

    finally:
        inferred = iteration - skipped
        if iteration > 0:
            print(f"\n── Summary ({iteration} iterations) ──")
            print(f"  Dark/skipped      : {skipped}")
            print(f"  Inferred          : {inferred}")
            print(f"  Retried           : {retried}")
            print(f"  Parse failures    : {parse_failures}")
            if inferred > 0:
                print(f"  Avg capture time  : {total_capture   / iteration:.2f}s")
                print(f"  Avg inference time: {total_inference / inferred:.2f}s")
                print(f"  Avg cycle time    : {(total_capture + total_inference) / iteration:.2f}s")


if __name__ == "__main__":
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    model, processor = load_model(model_name, device)
    run_pipeline(model, processor, device)
