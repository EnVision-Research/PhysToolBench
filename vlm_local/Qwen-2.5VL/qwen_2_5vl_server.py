from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import io
import argparse
import torch
import time

app = FastAPI(title="Qwen-2.5VL API")

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
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    # "Qwen/Qwen2.5-VL-72B-Instruct", 
    "Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype="auto", 
    device_map="auto"
)
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

model.to(device)

@app.post("/generate")
async def generate_response(
    image: UploadFile = File(...),
    prompt: str = Form("Describe this image.")
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
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)
    
    # Generate response
    generated_ids = model.generate(**inputs, max_new_tokens=1280)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    end_time = time.time()
    processing_time = end_time - start_time
    print(f"Request processed in {processing_time:.2f} seconds.")
    
    return {"response": output_text.strip()}

if __name__ == "__main__":
    # Set command line arguments
    parser = argparse.ArgumentParser(description='Qwen-2.5VL API Server')
    parser.add_argument('-p', '--port', type=int, default=8004, help='Port to run the server on (default: 8004)')
    parser.add_argument('-H', '--host', type=str, default="0.0.0.0", help='Host to run the server on (default: 0.0.0.0)')
    args = parser.parse_args()
    
    print(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)