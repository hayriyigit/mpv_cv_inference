# MPV Inference Sync Demo

This project demonstrates how to pair **libmpv** video decoding with a lightweight
machine-learning inference pipeline that runs on a server and streams results to
remote clients. Both the server and the client decode the same video source via
mpv; inference metadata, frame hashes, and timing information are exchanged over
ZeroMQ so that overlays remain visually aligned even when the client seeks or
pauses playback.

## Architecture

- **Inference server** (`src/server/inference_server.py`)
	- Uses mpv's `screenshot-raw` command with `hwdec=auto` to extract decoded
		frames without re-encoding.
	- Runs a sample brightness hotspot detector (extendable to arbitrary ML
		models) and publishes `InferenceResult` payloads over ZeroMQ `PUB`.
	- Exposes a ZeroMQ `REP` control channel for starting/stopping models and
		adjusting their parameters.
	- Every payload includes the mpv presentation timestamp, a Blake2b frame hash,
		a monotonic frame counter, and detection geometry for overlays.

- **Inference client** (`src/client/inference_client.py`)
	- Plays the same video via mpv, subscribes to inference updates, and caches
		recent payloads.
	- Recomputes the local frame hash on demand to assert synchronisation before
		drawing overlays.
	- Uses mpv's `ImageOverlay` helper to render bounding boxes and labels during
		playback, and redraws the cached inference data when the user seeks.
	- Provides a simple REPL for issuing control commands back to the server.

Shared Pydantic models live in `src/shared/models.py`, and lightweight tests
reside under `tests/`.

## Requirements

- Python >= 3.13
- System packages: `mpv`, `libmpv`, `libzmq`
- Optional but recommended: GPU drivers that support `hwdec=auto` in mpv.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install .[dev]
```

> **Note:** Ensure `mpv` is discoverable on your `$PATH` and the dynamic library
> can be loaded (e.g., `libmpv.so`, `mpv-2.dll`).

## Running the server

```bash
python -m src.server.inference_server \
	--video /path/to/video.mp4 \
	--pub tcp://0.0.0.0:5556 \
	--ctrl tcp://0.0.0.0:5557 \
	--sample-interval 0.125 \
	--hotspot-sensitivity 0.45 \
	--hotspot-min-pixels 1000
```

- `--sample-interval` controls how often frames are sampled for inference.
- `--hotspot-sensitivity` sets the brightness threshold (0-1). Lower values
	produce more detections and are useful for darker footage.
- `--hotspot-min-pixels` defines how many bright pixels must be present before a
	detection is emitted.
- Adjust `--pub`/`--ctrl` addresses to bind on the correct interfaces when the
	client runs on a different host.

## Running the client

```bash
python -m src.client.inference_client \
	--video /path/to/video.mp4 \
	--sub tcp://SERVER_HOST:5556 \
	--ctrl tcp://SERVER_HOST:5557 \
	--tolerance 0.05
```

Once started, the client REPL accepts:

- `status` – Request the current status of all models.
- `start <model>` / `stop <model>` – Enable or disable a model on the server.
- `set <model> key=value [...]` – Update model parameters (e.g.,
	`set brightness_hotspot sensitivity=0.6` to increase sensitivity on the fly).
- `quit` – Exit the client.

Overlays are redrawn automatically whenever the mpv playback time changes or
new inference results arrive that fall within the configured tolerance window.
If your GPU pipeline alters frames slightly, you may see "Frame hash mismatch"
warnings—when the timestamps still align (within ~2× tolerance) the client now
trusts the server hash and continues rendering overlays.

## Testing

```bash
pytest
```

The tests cover JSON round-tripping of shared models, control command handling,
and the client-side inference cache.

## Next steps

- Swap the sample brightness detector for production ML models (PyTorch,
	TensorFlow, ONNX, etc.).
- Replace the `screenshot-raw` path with the libmpv render API for higher
	throughput and GPU-to-GPU transfers.
- Add authentication and health monitoring around the ZeroMQ channels.
- Persist inference history to a database for later analytics.