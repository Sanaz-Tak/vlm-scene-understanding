# Real-Time Scene Understanding with Vision-Language Models

A webcam-based perception pipeline that uses [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) to continuously describe a live scene and flag potentially dangerous events (fights, fires, accidents, medical emergencies) — aimed at robotics perception and surveillance use cases.

**Status: early-stage / actively developed.** This is a solo research project exploring whether a small, locally-run VLM can do useful real-time scene understanding on consumer GPU hardware. It is not yet a finished system — see [Current Limitations](#current-limitations) below.

## Why

Most video anomaly/danger detection systems either rely on task-specific trained classifiers (fast, but narrow and require labeled data per event type) or cloud VLM APIs (flexible, but not privacy-preserving or real-time on-device). This project explores the middle ground: a small, quantized, open-weight VLM running entirely on a local GPU, producing structured, human-readable scene descriptions plus a lightweight danger classification — with an eye toward robotics perception and surveillance applications where on-device inference matters.

## How it works

1. **Capture** — grabs a short clip (default 3s) from a webcam via `ffmpeg`.
2. **Dark-frame filter** — skips clips with low mean brightness (e.g., camera physically covered or a dark room) before wasting a GPU inference pass on them.
3. **Inference** — feeds the clip to Qwen2.5-VL-3B-Instruct (4-bit quantized via `bitsandbytes`) with a structured prompt asking for a JSON object: `danger_detected`, `description`, `confidence`.
4. **Parsing + retry** — extracts the JSON from the model's raw output; if parsing fails, retries once with a simplified prompt.
5. **Loop** — repeats continuously, logging per-iteration timing (capture time, inference time) and a running summary (skip rate, retry rate, parse-failure rate, average cycle time).

## Hardware

Developed and tested on an **NVIDIA RTX 4060** (8GB VRAM), using 4-bit quantization to fit the 3B-parameter model comfortably. The same approach is intended to be portable to edge devices like the NVIDIA Jetson Orin for actual robotics deployment.

## Current limitations

This is being published in-progress, deliberately, as a snapshot of an ongoing research effort:

- **Inference latency (~15–55s per 3-second clip)** is far from real-time. The main driver is the number of frames the model processes per clip; reducing frame count via the model's native video-sampling controls is the next optimization target.
- **Structured JSON output is not fully reliable** at 4-bit quantization on a 3B model — the retry-with-simplified-prompt mechanism recovers most failures, but a more robust output strategy (e.g., constrained decoding, or decoupling scene description from a lightweight keyword/semantic danger classifier) is under evaluation.
- **No formal evaluation yet** against a labeled dataset (e.g., UCF-Crime, ShanghaiTech). Quantitative precision/recall numbers are a near-term goal.

## Setup

```bash
pip install -r requirements.txt
```

You'll also need `ffmpeg` installed and available on your `PATH`, and a webcam accessible to it. Update `DEVICE_NAME` in `vlm_pipeline.py` to match your camera (find yours with `ffmpeg -list_devices true -f dshow -i dummy` on Windows, or `v4l2-ctl --list-devices` on Linux).

```bash
python vlm_pipeline.py
```

## Roadmap

- [ ] Reduce per-clip frame count for faster inference (targeting sub-5s cycle time)
- [ ] Evaluate on a public anomaly-detection benchmark (UCF-Crime / ShanghaiTech)
- [ ] Port to NVIDIA Jetson Orin for embedded/robotics deployment
- [ ] Compare structured-JSON prompting vs. free-form description + separate classifier
- [ ] Write up results for submission to ICRA or IROS 2027

## Author

**Sanaz M. Takaghaj, PhD** — Assistant Teaching Professor, Penn State. [Portfolio](https://sanaz-tak.github.io/sanaz/) · [Google Scholar](https://scholar.google.com/citations?hl=en&user=IcEFxs4AAAAJ)

## License

MIT — see [LICENSE](LICENSE).
