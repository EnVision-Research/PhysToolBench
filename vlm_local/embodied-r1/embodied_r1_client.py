import requests
import argparse
import os
import json

def get_response(image_path, prompt, mode, url):
    """
    Sends a request to the Embodied-R1 server.
    """
    if not os.path.exists(image_path):
        return {"error": "Image file not found."}

    with open(image_path, 'rb') as f:
        files = {'image': (os.path.basename(image_path), f, 'image/jpeg')}
        data = {'prompt': prompt, 'mode': mode}
        try:
            response = requests.post(url, files=files, data=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Client for Embodied-R1 API Server')
    parser.add_argument('--image_path', type=str, default='exsample_data/put the red block on top of the yellow block.png', help='Path to the image file.')
    parser.add_argument('-P', '--prompt', type=str, default='put the red block on top of the yellow block', help='Prompt for the model.')
    parser.add_argument('-m', '--mode', type=str, default='VTG', choices=['REG', 'OFG', 'RRG', 'VTG'], help='Inference mode.')
    parser.add_argument('-p', '--port', type=int, default=8006, help='Port the server is running on.')
    parser.add_argument('-H', '--host', type=str, default='localhost', help='Host the server is running on.')
    
    args = parser.parse_args()
    
    api_url = f"http://{args.host}:{args.port}/generate"
    
    print(f"Sending request to {api_url} with image '{args.image_path}'")
    
    result = get_response(args.image_path, args.prompt, args.mode, api_url)
    
    print("\n--- Model Response ---")
    print(json.dumps(result, indent=2))
    print("----------------------\n")