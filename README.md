# Real-Time Scene Understanding with Vision-Language Models

**Status: early-stage / actively developed.** This is a solo research project exploring whether a small, locally-run VLM can do useful real-time scene understanding on consumer GPU hardware. It is not yet a finished system — see [Current Limitations](#current-limitations) below.

## Why

Most video anomaly/danger detection systems either rely on task-specific trained classifiers (fast, but narrow and require labeled data per event type) or cloud VLM APIs (flexible, but not privacy-preserving or real-time on-device). This project explores the middle ground: a small, quantized, open-weight VLM running entirely on a local GPU, producing structured, human-readable scene descriptions plus a lightweight danger classification — with an eye toward robotics perception and surveillance applications where on-device inference matters.

## How it works

1. **Capture** — grabs a short clip (default 3s) from a webcam via `ffmpeg`.
2. **Dark-frame filter** — skips clips with low mean brightness (e.g., camera physically covered or a dark room) before wasting a GPU inference pass on them.
3. **Inference** — feeds the clip to Qwen2.5-VL-3B-Instruct (4-bit quantized via `bitsandbytes`) with a structured prompt asking for a JSON object: `danger_detected`, `description`, `confidence`.
4. **Parsing + retry** — extracts the JSON from the model's raw output; if parsing fails, retries once with a simplified prompt.
5. **Loop** — repeats continuously, logging per-iteration timing (capture time, inference time) and a running summary (skip rate, retry rate, parse-failure rate, average cycle time).

## Current limitations

This is being published in-progress, deliberately, as a snapshot of an ongoing research effort:

- **Inference latency (~15–55s per 3-second clip)** is far from real-time. The main driver is the number of frames the model processes per clip; reducing frame count via the model's native video-sampling controls is the next optimization target.
- **Structured JSON output is not fully reliable** at 4-bit quantization on a 3B model — the retry-with-simplified-prompt mechanism recovers most failures, but a more robust output strategy (e.g., constrained decoding, or decoupling scene description from a lightweight keyword/semantic danger classifier) is under evaluation.
- **No formal evaluation yet** against a labeled dataset (e.g., UCF-Crime, ShanghaiTech). Quantitative precision/recall numbers are a near-term goal.

## Setup

```bash
pip install -r requirements.txt
```

You'll also need `ffmpeg` installed and available on your `PATH`, and a webcam accessible to it. Set the `CAMERA_DEVICE` environment variable to match your camera:

```bash
# Windows — find your device name with: ffmpeg -list_devices true -f dshow -i dummy
set CAMERA_DEVICE=HD Webcam

# Linux — typically /dev/video0
export CAMERA_DEVICE=/dev/video0

# macOS — find your device index with: ffmpeg -f avfoundation -list_devices true -i ""
export CAMERA_DEVICE=0
```

```bash
python vlm_pipeline.py
```

## Author

**Sanaz M. Takaghaj, PhD** — Assistant Teaching Professor, Penn State. [Portfolio](https://sanaz-tak.github.io/sanaz/) · [Google Scholar](https://scholar.google.com/citations?hl=en&user=IcEFxs4AAAAJ)

## License

MIT — see [LICENSE](LICENSE).
