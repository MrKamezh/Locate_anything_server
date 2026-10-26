import io
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from worker import LocateAnythingWorker

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

worker = None

TASK_ALIASES = {
    "detection": "Detection",
    "grounding": "Grounding",
    "ocr": "OCR",
    "orc": "OCR",
    "gui": "GUI",
    "pointing": "Pointing",
}


def get_worker() -> LocateAnythingWorker:
    global worker
    if worker is None:
        worker = LocateAnythingWorker()
    return worker


def normalize_task_type(task_type: str) -> str:
    return TASK_ALIASES.get(task_type.strip().lower(), "Detection")


def build_prompt(prompt: str, task_type: str) -> str:
    cleaned_prompt = prompt.strip()

    if task_type == "Detection":
        categories = [item.strip() for item in cleaned_prompt.split(",") if item.strip()]
        category_prompt = "</c>".join(categories) if categories else cleaned_prompt
        return (
            "Detect every visible instance for the requested object categories. "
            f"Categories: {category_prompt}."
        )

    if task_type == "Grounding":
        return (
            "Ground this referring expression and return only the single best matching region: "
            f"{cleaned_prompt}."
        )

    if task_type == "OCR":
        if cleaned_prompt:
            return f"Please locate the text referred as {cleaned_prompt}."
        return "Detect all the text in box format."

    if task_type == "GUI":
        return f"Locate the region that matches the following description: {cleaned_prompt}."

    if task_type == "Pointing":
        return f"Point to: {cleaned_prompt}."

    return cleaned_prompt


@app.post("/process-frame")
async def process_frame(file: UploadFile = File(...), prompt: str = Form(...), task_type: str = Form("Detection")):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    original_width, original_height = image.size

    # Resize to 512x512 to keep VRAM usage low
    image = image.convert("RGB").resize((512, 512))

    task_type = normalize_task_type(task_type)
    final_prompt = build_prompt(prompt, task_type)

    model_worker = get_worker()
    raw_text = model_worker.predict(image, final_prompt, generation_mode="hybrid")
    pixel_boxes = model_worker.parse_boxes(raw_text, original_width, original_height)
    pixel_points = model_worker.parse_points(raw_text, original_width, original_height)
    if task_type == "Grounding":
        pixel_boxes = pixel_boxes[:1]
        pixel_points = pixel_points[:1]
    return {
        "task_type": task_type,
        "prompt": final_prompt,
        "boxes": pixel_boxes,
        "points": pixel_points,
        "raw_output": raw_text,
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
