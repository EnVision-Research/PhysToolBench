import torch
import requests
from PIL import Image
from transformers import AutoModelForCausalLM

MODEL_PATH = "AIDC-AI/Ovis2.5-9B"

# Thinking mode & budget
enable_thinking = True
enable_thinking_budget = True  # Only effective if enable_thinking is True.

# Total tokens for thinking + answer. Ensure: max_new_tokens > thinking_budget + 25
max_new_tokens = 3072
thinking_budget = 2048

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).cuda()

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": Image.open(requests.get("https://cdn-uploads.huggingface.co/production/uploads/658a8a837959448ef5500ce5/TIlymOb86R6_Mez3bpmcB.png", stream=True).raw)},
        {"type": "text", "text": "Calculate the sum of the numbers in the middle box in figure (c)."},
    ],
}]

input_ids, pixel_values, grid_thws = model.preprocess_inputs(
    messages=messages,
    add_generation_prompt=True,
    enable_thinking=enable_thinking
)
input_ids = input_ids.cuda()
pixel_values = pixel_values.cuda() if pixel_values is not None else None
grid_thws = grid_thws.cuda() if grid_thws is not None else None

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
print(response)
