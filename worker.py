import re
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor

class LocateAnythingWorker:
    def __init__(self, model_path: str = "nvidia/LocateAnything-3B", device: str = "cuda", dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype
        print(f"Loading {model_path} on {device} with dtype {dtype}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path, 
            torch_dtype=dtype, 
            trust_remote_code=True,
            attn_implementation="sdpa" # Scaled Dot Product Attention
        ).to(device).eval()
        print("Model loaded successfully on your GPU.")

    @torch.no_grad()
    def predict(self, image: Image.Image, question: str, generation_mode: str = "hybrid") -> str:
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]}]
        text = self.processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(text=[text], images=images, videos=videos, return_tensors="pt").to(self.device)
        
        response = self.model.generate(
            pixel_values=inputs["pixel_values"].to(self.dtype),
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_grid_hws=inputs.get("image_grid_hws", None),
            tokenizer=self.tokenizer,
            max_new_tokens=2048,
            use_cache=True,
            generation_mode=generation_mode,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            verbose=False,
        )
        return response[0] if isinstance(response, tuple) else response

    @staticmethod
    def parse_boxes(answer: str, image_width: int, image_height: int) -> list[dict]:
        boxes = []
        for m in re.finditer(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer):
            x1, y1, x2, y2 = [int(g) for g in m.groups()]
            boxes.append({
                "x1": (x1 / 1000) * image_width, "y1": (y1 / 1000) * image_height,
                "x2": (x2 / 1000) * image_width, "y2": (y2 / 1000) * image_height,
            })
        return boxes

    @staticmethod
    def parse_points(answer: str, image_width: int, image_height: int) -> list[dict]:
        points = []
        point_pattern = r"<(?:box|point)><(\d+)><(\d+)></(?:box|point)>"
        for m in re.finditer(point_pattern, answer):
            x, y = [int(g) for g in m.groups()]
            points.append({
                "x": (x / 1000) * image_width,
                "y": (y / 1000) * image_height,
            })
        return points
