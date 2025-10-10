from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
import io
import argparse
import torch
import time

app = FastAPI(title="Gemma3 API")

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

# Load model and processor globally
model_id = "google/gemma-3-27b-it"

model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype="auto", 
    device_map="auto"
).eval()
print("Model loaded")
processor = AutoProcessor.from_pretrained(model_id)

@app.post("/generate")
async def generate_response(
    image: UploadFile = File(...),
    prompt: str = Form("Describe this image in detail.")
):
    start_time = time.time()
    # Read and process the uploaded image
    image_content = await image.read()
    raw_image = Image.open(io.BytesIO(image_content))
    
    # Prepare messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": raw_image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    
    # Process inputs
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=model.dtype)
    
    input_len = inputs["input_ids"].shape[-1]
    
    # Generate response
    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        generation = generation[0][input_len:]

    decoded = processor.decode(generation, skip_special_tokens=True)
    
    end_time = time.time()
    processing_time = end_time - start_time
    print(f"Request processed in {processing_time:.2f} seconds.")
    
    return {"response": decoded.strip()}

if __name__ == "__main__":
    # Set command line arguments
    parser = argparse.ArgumentParser(description='Gemma3 API Server')
    parser.add_argument('-p', '--port', type=int, default=8003, help='Port to run the server on (default: 8003)')
    parser.add_argument('-H', '--host', type=str, default="0.0.0.0", help='Host to run the server on (default: 0.0.0.0)')
    args = parser.parse_args()
    
    print(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
