import requests
import argparse
import os

def get_response(image_path, prompt, url):
    """
    Sends a request to the RoboBrain2.0 server with an image and a prompt.
    """
    if not os.path.exists(image_path):
        return {"error": "Image file not found."}

    with open(image_path, 'rb') as f:
        files = {'image': (os.path.basename(image_path), f, 'image/jpeg')}
        data = {'prompt': prompt}
        try:
            response = requests.post(url, files=files, data=data)
            response.raise_for_status()  # Raise an exception for bad status codes
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Client for RoboBrain2.0 API Server')
    parser.add_argument('--image_path', type=str, default='http://images.cocodataset.org/val2017/000000039769.jpg', help='Path or URL to the image file.')
    parser.add_argument('-P', '--prompt', type=str, default='Describe this image in detail.', help='Prompt for the model. Default: "Describe this image in detail."')
    parser.add_argument('-p', '--port', type=int, default=8005, help='Port the server is running on. Default: 8005')
    parser.add_argument('-H', '--host', type=str, default='localhost', help='Host the server is running on. Default: "localhost"')
    
    args = parser.parse_args()
    
    image_path = args.image_path
    
    # If the image_path is a URL, download it first
    if image_path.startswith('http'):
        try:
            img_data = requests.get(image_path).content
            temp_image_path = "temp_downloaded_image.jpg"
            with open(temp_image_path, 'wb') as handler:
                handler.write(img_data)
            image_path = temp_image_path
        except requests.exceptions.RequestException as e:
            print(f"Error downloading image: {e}")
            exit()

    api_url = f"http://{args.host}:{args.port}/generate"
    
    print(f"Sending request to {api_url} with image '{args.image_path}'")
    
    result = get_response(image_path, args.prompt, api_url)

    if image_path.startswith('temp_downloaded_image'):
        os.remove(image_path)
    
    if "response" in result:
        print("\n--- Model Response ---")
        print(result["response"])
        print("----------------------\n")
    else:
        print("\n--- Error ---")
        print(result.get("error", "An unknown error occurred."))
        if "message" in result:
            print(result["message"])
        print("-------------\n")