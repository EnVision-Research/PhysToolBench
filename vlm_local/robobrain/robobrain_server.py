from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from PIL import Image
import io
import argparse
import time
import os
import uuid

# This is in the same directory.
from inference import UnifiedInference

app = FastAPI(title="RoboBrain2.0 API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model globally
model = UnifiedInference("BAAI/RoboBrain2.0-7B")

@app.post("/generate")
async def generate_response(
    image: UploadFile = File(...),
    prompt: str = Form("Describe this image.")
):
    start_time = time.time()
    
    # Save the uploaded file temporarily
    temp_dir = "temp_images"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create a unique filename
    file_extension = image.filename.split('.')[-1]
    unique_filename = f"{uuid.uuid4()}.{file_extension}"
    image_path = os.path.join(temp_dir, unique_filename)
    
    with open(image_path, "wb") as buffer:
        buffer.write(await image.read())
        
    # Perform inference using the UnifiedInference class
    # The `inference` method expects a file path.
    pred = model.inference(prompt, image_path, task="general", enable_thinking=True, do_sample=True)
    
    # Clean up the temporary file
    os.remove(image_path)

    end_time = time.time()
    processing_time = end_time - start_time
    print(f"Request processed in {processing_time:.2f} seconds.")
    
    return {"response": pred}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RoboBrain2.0 API Server')
    parser.add_argument('-p', '--port', type=int, default=8088, help='Port to run the server on (default: 8005)')
    parser.add_argument('-H', '--host', type=str, default="0.0.0.0", help='Host to run the server on (default: 0.0.0.0)')
    args = parser.parse_args()
    
    print(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)