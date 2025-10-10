from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch
from PIL import Image
from transformers import AutoModelForCausalLM
import io
import argparse
import time

app = FastAPI(title="Ovis2.5 API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model globally
MODEL_PATH = "AIDC-AI/Ovis2.5-9B"

enable_thinking = True
enable_thinking_budget = True
max_new_tokens = 3072
thinking_budget = 2048

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).to(device).eval()
print("Model loaded")

@app.post("/generate")
async def generate_response(
    image: UploadFile = File(...),
    prompt: str = Form("Describe this image in detail.")
):
    start_time = time.time()
    # Read and process the uploaded image
    image_content = await image.read()
    raw_image = Image.open(io.BytesIO(image_content))

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": raw_image},
            {"type": "text", "text": prompt},
        ],
    }]

    input_ids, pixel_values, grid_thws = model.preprocess_inputs(
        messages=messages,
        add_generation_prompt=True,
        enable_thinking=enable_thinking
    )
    input_ids = input_ids.to(device)
    pixel_values = pixel_values.to(device) if pixel_values is not None else None
    grid_thws = grid_thws.to(device) if grid_thws is not None else None

    with torch.inference_mode():
        outputs = model.generate(
            inputs=input_ids,
            pixel_values=pixel_values,
            grid_thws=grid_thws,
            enable_thinking=enable_thinking,
            enable_thinking_budget=enable_thinking_budget,
            max_new_tokens=max_new_tokens,
            thinking_budget=thinking_budget,
        )

        response = model.text_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    end_time = time.time()
    processing_time = end_time - start_time
    print(f"Request processed in {processing_time:.2f} seconds.")
    
    return {"response": response.strip()}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ovis2.5 API Server')
    parser.add_argument('-p', '--port', type=int, default=8011, help='Port to run the server on (default: 8002)')
    parser.add_argument('-H', '--host', type=str, default="0.0.0.0", help='Host to run the server on (default: 0.0.0.0)')
    args = parser.parse_args()
    
    print(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)