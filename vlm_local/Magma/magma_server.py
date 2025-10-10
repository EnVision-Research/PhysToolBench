from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import io
import argparse
import torch
import time

app = FastAPI(title="Magma API")

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
dtype = torch.bfloat16
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Magma-8B", 
    trust_remote_code=True, 
    torch_dtype=dtype
)
processor = AutoProcessor.from_pretrained(
    "microsoft/Magma-8B", 
    trust_remote_code=True
)

model.to(device)

@app.post("/generate")
async def generate_response(
    image: UploadFile = File(...),
    prompt: str = Form("What is in this image?")
):
    start_time = time.time()
    # Read and process the uploaded image
    image_content = await image.read()
    raw_image = Image.open(io.BytesIO(image_content))
    raw_image = raw_image.convert("RGB")
    
    # Prepare conversation messages similar to Magma's format
    convs = [
        # {"role": "system", "content": "You are agent that can see, talk and act."},
        {"role": "user", "content": f"<image_start><image><image_end>\n{prompt}"},
    ]
    
    # Process inputs using Magma's format
    prompt_text = processor.tokenizer.apply_chat_template(convs, tokenize=False, add_generation_prompt=True)
    inputs = processor(images=[raw_image], texts=prompt_text, return_tensors="pt")
    inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)
    inputs['image_sizes'] = inputs['image_sizes'].unsqueeze(0)
    inputs = inputs.to(device).to(dtype)
    
    # Generation arguments
    generation_args = { 
        "max_new_tokens": 1024, 
        "temperature": 0.5, 
        "do_sample": False, 
        "use_cache": True,
        "num_beams": 1,
    }
    
    # Generate response
    with torch.inference_mode():
        generate_ids = model.generate(**inputs, **generation_args)
    
    generate_ids = generate_ids[:, inputs["input_ids"].shape[-1] :]
    output_text = processor.decode(generate_ids[0], skip_special_tokens=True).strip()

    end_time = time.time()
    processing_time = end_time - start_time
    print(f"Request processed in {processing_time:.2f} seconds.")
    
    return {"response": output_text}

if __name__ == "__main__":
    # Set command line arguments
    parser = argparse.ArgumentParser(description='Magma API Server')
    parser.add_argument('-p', '--port', type=int, default=8005, help='Port to run the server on (default: 8005)')
    parser.add_argument('-H', '--host', type=str, default="0.0.0.0", help='Host to run the server on (default: 0.0.0.0)')
    args = parser.parse_args()
    
    print(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)