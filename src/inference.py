from model_api import generate_response
import os
import json
import sys
import argparse
import concurrent.futures
from functools import partial
from tqdm import tqdm

# from dataset.dataset_reader import DatasetReader
from datasets import load_dataset

def process_sample(sample_id, *, model_name, dataset, api_url, api_key, output_dir, mode, resume):

    sample = dataset['train'][sample_id]
    cot_prompt, no_cot_prompt = sample['cot_prompt'], sample['no_cot_prompt']
    correct_tool = sample['correct_tool']
    image = sample['image']
    difficulty = sample['difficulty']
    id = sample['id'] # id with task difficulty and id, e.g.'Easy_0001'

    output_filepath = os.path.join(output_dir, f"{str(id)}.json")
    if resume and os.path.exists(output_filepath):
        print(f"Result for sample {sample_id+1} in {output_dir} already exists. Skipping.")
        return

    cot_response = ""
    cot_answer = ""
    no_cot_answer = ""

    if mode in ["both", "cot_only"]:
        cot_response = generate_response(
            model_name,
            cot_prompt,
            image,
            api_url,
            api_key
        )
        if "### Answer" in cot_response:
            cot_answer = cot_response.split("### Answer")[1].split("\'\'\'")[0].replace('```', '').replace('\n', '')
    
    if mode in ["both", "no_cot_only"]:
        no_cot_answer = generate_response(
            model_name,
            no_cot_prompt,
            image,
            api_url,
            api_key
        )

    print(f"Current Task ID: {id}")
    print(f"COT Answer: {cot_answer}")
    print(f"No COT Answer: {no_cot_answer}")
    print(f"Correct Tool: {correct_tool}")
    print('-'*50)

    with open(output_filepath, "w") as f:
        json.dump({
            "cot_response_full": cot_response,
            "cot_answer": cot_answer,
            "no_cot_answer": no_cot_answer,
            "correct_tool": correct_tool,
            "difficulty": difficulty
        }, f, indent=4)

def inference_all(model_name, dataset, api_url, api_key, mode="both", resume=False, num_threads=1):
    output_dir = f"results/{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    inference(model_name, dataset, api_url, api_key, output_dir, mode, resume, num_threads)


def inference(model_name, dataset, api_url, api_key, output_dir, mode="both", resume=False, num_threads=1):
    total_samples = dataset['train'].num_rows
    if num_threads > 1:
        process_func = partial(process_sample, model_name=model_name, dataset=dataset, api_url=api_url, api_key=api_key, output_dir=output_dir, mode=mode, resume=resume)
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            list(tqdm(executor.map(process_func, range(total_samples)), total=total_samples, desc=f"Inferencing"))
    else:
        for sample_id in tqdm(range(total_samples), total=total_samples, desc=f"Inferencing"):
            process_sample(sample_id, model_name=model_name, dataset=dataset, api_url=api_url, api_key=api_key, output_dir=output_dir, mode=mode, resume=resume)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gemma-3")
    parser.add_argument("--api_url", type=str, default="http://localhost:8003")
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--mode", type=str, default="both", choices=["both", "cot_only", "no_cot_only"], help="Inference mode. 'both' runs CoT and no-CoT, 'cot_only' runs only CoT, 'no_cot_only' runs only no-CoT.")
    parser.add_argument("--resume", default=False, action="store_true", help="Resume inference, skip existing results.")
    parser.add_argument("--num_threads", type=int, default=1, help="Number of threads for parallel inference.")
    args = parser.parse_args()

    dataset = load_dataset("zhangzixin02/PhysToolBench")
    
    inference_all(args.model_name, dataset, args.api_url, args.api_key, args.mode, args.resume, args.num_threads)