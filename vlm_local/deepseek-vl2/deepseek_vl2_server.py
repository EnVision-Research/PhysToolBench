import os
# Set tokenizer parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from PIL import Image
import io
import argparse
import torch
import time
from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor
from transformers import AutoModelForCausalLM

app = FastAPI(title="DeepSeek-VL2 API")

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
# model_path = '/hpc2hdd/home/zzhang300/.cache/huggingface/hub/models--deepseek-ai--deepseek-vl2-tiny/snapshots/66c54660eae7e90c9ba259bfdf92d07d6e3ce8aa'

# model_path = '/hpc2hdd/home/zzhang300/.cache/huggingface/hub/models--deepseek-ai--deepseek-vl2-small/snapshots/6033e16432a1d771cf9fe4a6f894ff5e5e1459af'

model_path = '/hpc2hdd/home/zzhang300/.cache/huggingface/hub/models--deepseek-ai--deepseek-vl2/snapshots/f363772d1c47f4239dd844015b4bd53beb87951b'

processor = DeepseekVLV2Processor.from_pretrained(model_path)
tokenizer = processor.tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
).to(device).eval()


@app.post("/generate")
async def generate_response(
    image: UploadFile = File(...),
    prompt: str = Form("Describe this image.")
):
    start_time = time.time()
    # Read and process the uploaded image
    image_content = await image.read()
    pil_image = Image.open(io.BytesIO(image_content)).convert("RGB")
    
    # Prepare messages
    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image>\n{prompt}",
            "images": ["placeholder"], # placeholder for processor
        },
        {"role": "<|Assistant|>", "content": ""},
    ]
    
    # Process inputs
    prepare_inputs = processor(
        conversations=conversation,
        images=[pil_image],
        force_batchify=True,
        system_prompt=""
    ).to(model.device, dtype=torch.bfloat16)
    
    # Generate response
    with torch.no_grad():
        # Pass the expected attributes directly like in the inference.py
        outputs = model.generate(
            input_ids=prepare_inputs.input_ids,
            images=prepare_inputs.images,
            images_seq_mask=prepare_inputs.images_seq_mask,
            images_spatial_crop=prepare_inputs.images_spatial_crop,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.4,
            top_p=0.9,
            repetition_penalty=1.1,
            use_cache=True,
        )
        
    answer = tokenizer.decode(outputs[0][len(prepare_inputs.input_ids[0]):].cpu().tolist(), skip_special_tokens=True)

    end_time = time.time()
    processing_time = end_time - start_time
    print(f"Request processed in {processing_time:.2f} seconds.")
    
    return {"response": answer.strip()}

if __name__ == "__main__":
    # Set command line arguments
    parser = argparse.ArgumentParser(description='DeepSeek-VL2 API Server')
    parser.add_argument('-p', '--port', type=int, default=8016, help='Port to run the server on (default: 8005)')
    parser.add_argument('-H', '--host', type=str, default="0.0.0.0", help='Host to run the server on (default: 0.0.0.0)')
    parser.add_argument('--model_path', type=str, default=model_path, help='Path to the DeepSeek-VL2 model')
    args = parser.parse_args()
    
    # This is a bit of a hack to reload the model if a different path is provided
    if args.model_path != model_path:
        model_path = args.model_path
        print(f"Loading model from new path: {model_path}")
        processor = DeepseekVLV2Processor.from_pretrained(model_path)
        tokenizer = processor.tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        ).to(device).eval()

    print(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)