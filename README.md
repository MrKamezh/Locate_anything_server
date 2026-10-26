# LocateAnything Server

FastAPI backend that runs [nvidia/LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B) on a local GPU and returns bounding boxes / points for a natural-language prompt.

The web UI lives in a separate repo: `Locate_anything_interface`.

## Requirements

- Python 3.12 (tested with 3.12.9)
- NVIDIA GPU with ~8 GB+ VRAM and a CUDA 12.x driver
- ~7 GB disk space for the model (downloaded automatically from Hugging Face on first request)

## Setup

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` installs the CUDA 12.4 build of PyTorch. If you use a different CUDA version, install torch first using the command from [pytorch.org](https://pytorch.org/get-started/locally/).

## Run

```cmd
python main.py
```

The server starts on `http://localhost:8000`. The model loads on the first request, so that call takes a while. Interactive API docs: `http://localhost:8000/docs`.

## API

### `POST /process-frame`

`multipart/form-data` fields:

| Field       | Type   | Description |
|-------------|--------|-------------|
| `file`      | image  | Image to analyze |
| `prompt`    | string | Natural-language query |
| `task_type` | string | `Detection` (default), `Grounding`, `OCR`, `GUI`, `Pointing` |

Task behavior:

- **Detection**: comma-separated categories, returns every instance (`person, car`)
- **Grounding**: referring expression, returns the single best match
- **OCR**: locate specific text, or leave prompt empty to detect all text
- **GUI**: locate a UI element by description
- **Pointing**: returns coordinate points

Response:

```json
{
  "task_type": "Detection",
  "prompt": "<final prompt sent to the model>",
  "boxes":  [{ "x1": 10.0, "y1": 20.0, "x2": 110.0, "y2": 220.0 }],
  "points": [{ "x": 60.0, "y": 120.0 }],
  "raw_output": "<raw model text>"
}
```

Coordinates are in pixels of the original uploaded image.

Example:

```cmd
curl -X POST http://localhost:8000/process-frame -F "file=@photo.jpg" -F "prompt=person, dog" -F "task_type=Detection"
```

## Files

- `main.py`: FastAPI app, prompt building per task type
- `worker.py`: model loading, inference, and parsing of `<box>` / `<point>` output tokens

## Notes

- The API has no authentication and CORS allows all origins. Don't expose it publicly as-is.
- Model weights are not included in this repo. Check the model license on its Hugging Face page before any commercial use.
# Locate_anything_server
