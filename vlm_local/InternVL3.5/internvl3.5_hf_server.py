import torch
from fastapi import FastAPI, File, UploadFile, Form
from transformers import pipeline
import uvicorn
import argparse
from PIL import Image
import io

# Initialize FastAPI app
app = FastAPI()

# model_name = "OpenGVLab/InternVL3_5-14B-HF"
model_name = 'OpenGVLab/InternVL3_5-241B-A28B-HF'


# Load the model directly
pipe = pipeline("image-text-to-text", model=model_name, torch_dtype=torch.bfloat16, device_map="auto")

# Define the inference endpoint
@app.post("/generate")
async def generate(
    image: UploadFile = File(...),
    prompt: str = Form("<image>\nPlease describe the image shortly."),
    max_new_tokens: int = Form(1024),
    return_full_text: bool = Form(False)
):
    image_content = await image.read()
    pil_image = Image.open(io.BytesIO(image_content)).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    outputs = pipe(text=messages, max_new_tokens=max_new_tokens, return_full_text=return_full_text)
    response = outputs[0]["generated_text"]
    return {"response": response.strip()}

# Add a main block to run the server
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8021)
    args = parser.parse_args()
    
    uvicorn.run(app, host=args.host, port=args.port)