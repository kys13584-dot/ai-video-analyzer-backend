import urllib.request
import json
import time

url = "http://localhost:8000/api/videos"
data = json.dumps({"url": "https://www.youtube.com/watch?v=jNQXAC9IVRw"}).encode("utf-8")
req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")

try:
    with urllib.request.urlopen(req) as res:
        response_data = json.loads(res.read().decode('utf-8'))
        print("Initial response:", json.dumps(response_data, indent=2))
        video_id = response_data.get("id")
        
        if video_id:
            print(f"Polling status for video_id: {video_id}...")
            # Poll for completion
            while True:
                time.sleep(3)
                get_req = urllib.request.Request(f"{url}/{video_id}", method="GET")
                with urllib.request.urlopen(get_req) as get_res:
                    status_data = json.loads(get_res.read().decode("utf-8"))
                    print("Current status:", status_data.get("status"))
                    if status_data.get("status") in ["completed", "failed"]:
                        print("Final result:")
                        print(json.dumps(status_data, indent=2))
                        break
except Exception as e:
    print("Request failed:", e)
