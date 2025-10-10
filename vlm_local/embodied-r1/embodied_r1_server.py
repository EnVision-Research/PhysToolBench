from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from PIL import Image
import io
import argparse
import time
import re
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import os

# Set HF_ENDPOINT
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# --- Logic from inference_example.py ---

def process_vision_info(messages):
    image_inputs = []
    for msg in messages:
        for content in msg["content"] if isinstance(msg["content"], list) else [msg["content"]]:
            if isinstance(content, dict) and content.get("type") == "image":
                image_inputs.append(content["image"])
    return image_inputs, None

CONF_MODE = {
    "REG": {
        "template": (
            "Provide one or more points coordinate of objects region {instruction}. "
            "The results are presented in a format <point>[[x1,y1], [x2,y2], ...]</point>. "
            "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
            "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags. "
            "The answer consists only of several coordinate points, with the overall format being: "
            "<think> reasoning process here </think><answer><point>[[x1, y1], [x2, y2], ...]</point></answer>"
        ),
        "description": "Referring Expression Grounding - Locating the coordinates of specified object regions within an image."
    },
    "OFG": {
        "template": (
            "Please provide the 2D points coordinate of the region this sentence describes: {instruction}. "
            "The results are presented in a format <point>[[x1,y1], [x2,y2], ...]</point>. "
            "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
            "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags. "
            "The answer consists only of several coordinate points, with the overall format being: "
            "<think> reasoning process here </think><answer><point>[[x1, y1], [x2, y2], ...]</point></answer>"
        ),
        "description": "Object Affordance Grounding - Locating the 2D coordinates of specified object regions based on descriptions."
    },
    "RRG": {
        "template": (
            "You are currently a robot performing robotic manipulation tasks. The task instruction is: {instruction}. "
            "Use 2D points to mark the target location where the object you need to manipulate in the task should ultimately be moved. "
            "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
            "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags. "
            "The answer consists only of several coordinate points, with the overall format being: "
            "<think> reasoning process here </think><answer><point>[[x1, y1], [x2, y2], ...]</point></answer>"
        ),
        "description": "Region Referring Grounding - Locating specific spatial target locations for robotic manipulation tasks."
    },
    "VTG": {
        "template": (
            "You are currently a robot performing robotic manipulation tasks. The task instruction is: {instruction}. "
            "Use 2D points to mark the manipulated object-centric waypoints to guide the robot to successfully complete the task. "
            "You must provide the points in the order of the trajectory, and the number of points must be 8. "
            "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
            "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags. "
            "The answer consists only of several coordinate points, with the overall format being: "
            "<think> reasoning process here </think><answer><point>[[x1, y1], [x2, y2], ..., [x8, y8]]</point></answer>."
        ),
        "description": "Visual Trace Generation - Generating waypoints centered on manipulated objects to guide robots to complete tasks."
    }
}

def _load_model_processor(checkpoint_path, cpu_only=False):
    if cpu_only:
        device_map = 'cpu'
    else:
        device_map = 'auto'

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        checkpoint_path, 
        torch_dtype="auto",
        device_map=device_map
    )
    processor = AutoProcessor.from_pretrained(checkpoint_path)
    return model, processor

def _extract_model_output_parts(text):
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    think_content = think_match.group(1).strip() if think_match else ""
    
    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    answer_content = answer_match.group(1).strip() if answer_match else ""
    
    point_match = re.search(r'<point>(.*?)</point>', answer_content, re.DOTALL)
    coordinates = point_match.group(1).strip() if point_match else ""
    
    return think_content, answer_content, coordinates

def _transform_messages(original_messages):
    transformed_messages = []
    for message in original_messages:
        new_content = []
        for item in message['content']:
            if 'image' in item:
                new_item = {'type': 'image', 'image': item['image']}
            elif 'text' in item:
                new_item = {'type': 'text', 'text': item['text']}
            else:
                continue
            new_content.append(new_item)
        new_message = {'role': message['role'], 'content': new_content}
        transformed_messages.append(new_message)
    return transformed_messages

def run_inference(model, processor, messages):
    messages = _transform_messages(messages)
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors='pt')
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=4096,
            temperature=0,
            top_p=1,
            repetition_penalty=1.05,
            do_sample=False
        )
    
    generated_text = processor.batch_decode(
        generated_ids[:, inputs['input_ids'].shape[1]:], 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )[0]
    
    return generated_text

# --- FastAPI Server ---

app = FastAPI(title="Embodied-R1 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading model and processor...")
# DEFAULT_CKPT_PATH = 'IffYuan/Embodied-R1-3B-v1'
DEFAULT_CKPT_PATH = '~/.cache/huggingface/hub/models--IffYuan--Embodied-R1-3B-v1/snapshots/e6b6629aca99bc6b50bd565a46617379defe6c87'
model, processor = _load_model_processor(DEFAULT_CKPT_PATH)
print("Model loaded successfully")


@app.post("/generate")
async def generate_response_endpoint(
    image: UploadFile = File(...),
    prompt: str = Form("put the red block on top of the yellow block"),
    # mode: str = Form("VTG")
):
    start_time = time.time()
    
    image_content = await image.read()
    raw_image = Image.open(io.BytesIO(image_content)).convert("RGB")

    # if mode not in CONF_MODE:
    #     return {"error": f"Invalid mode '{mode}'. Supported modes are {list(CONF_MODE.keys())}"}

    # template = CONF_MODE[mode]["template"]
    # formatted_text = template.format(instruction=prompt)
    formatted_text = prompt

    messages = [{
        'role': 'user',
        'content': [
            {'type': 'image', 'image': raw_image},
            {'type': 'text', 'text': formatted_text}
        ]
    }]
    
    response_text = run_inference(model, processor, messages)
    return {'response': response_text}
    think_content, answer_content, coordinates = _extract_model_output_parts(response_text)

    end_time = time.time()
    processing_time = end_time - start_time
    print(f"Request processed in {processing_time:.2f} seconds.")

    return {
        "think_content": think_content,
        "answer_content": answer_content,
        "coordinates": coordinates,
        "full_response": response_text,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Embodied-R1 API Server')
    parser.add_argument('-p', '--port', type=int, default=8006, help='Port to run the server on (default: 8006)')
    parser.add_argument('-H', '--host', type=str, default="0.0.0.0", help='Host to run the server on (default: 0.0.0.0)')
    args = parser.parse_args()
    
    print(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)