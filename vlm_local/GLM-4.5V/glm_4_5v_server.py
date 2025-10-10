from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from transformers import AutoProcessor, Glm4vMoeForConditionalGeneration
from PIL import Image
import io
import argparse
import torch
import time

app = FastAPI(title="GLM-4.5V API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Check if GPU is available
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs.")
    if num_gpus > 1:
        print("Using `device_map='auto'` for multi-GPU inference.")
else:
    print("No GPU found, using CPU.")

# Load model and processor globally
MODEL_PATH = "zai-org/GLM-4.5V"
print(f"Loading processor from {MODEL_PATH}...")
processor = AutoProcessor.from_pretrained(MODEL_PATH)
print("Processor loaded.")

print("Loading model... This may take a while for large models.")
model = Glm4vMoeForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path=MODEL_PATH,
    torch_dtype="auto",
    device_map="auto",
)
print("Model loaded.")

# Print the device map to confirm multi-GPU usage
if hasattr(model, 'hf_device_map'):
    print("Model device map:")
    print(model.hf_device_map)
else:
    print("`hf_device_map` attribute not found on the model. Cannot print device map.")


@app.post("/generate")
async def generate_response(
    image: UploadFile = File(...),
    prompt: str = Form("describe this image")
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
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)
    inputs.pop("token_type_ids", None)
    
    # Generate response
    generated_ids = model.generate(**inputs, max_new_tokens=8192)
    output_text = processor.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    end_time = time.time()
    processing_time = end_time - start_time
    print(f"Request processed in {processing_time:.2f} seconds.")
    
    return {"response": output_text.strip()}

if __name__ == "__main__":
    # Set command line arguments
    parser = argparse.ArgumentParser(description='GLM-4.5V API Server')
    parser.add_argument('-p', '--port', type=int, default=8005, help='Port to run the server on (default: 8005)')
    parser.add_argument('-H', '--host', type=str, default="0.0.0.0", help='Host to run the server on (default: 0.0.0.0)')
    args = parser.parse_args()
    
    print(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)