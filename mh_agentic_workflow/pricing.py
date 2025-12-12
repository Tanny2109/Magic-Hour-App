import requests
from dotenv import load_dotenv
import os
load_dotenv()
FAL_KEY=os.getenv("FAL_KEY")

url = "https://api.fal.ai/v1/models/pricing"

# querystring = {"endpoint_id":"\"fal-ai/bytedance/seedance/v1/lite/text-to-video\",         #                    \"fal-ai/bytedance/seedance/v1/pro/text-to-video\",         #                    \"fal-ai/bytedance/seedance/v1/pro/fast/text-to-video\",         #                    \"fal-ai/ltxv-2/text-to-video/fast\",         #                    \"fal-ai/ltxv-2/text-to-video/\""}

headers = {"Authorization":"55d4cca6-814f-47ba-8e80-88ff277fd07e:0acd86396a06b3f63d75cdaa3a5bb8f2"}

response = requests.get(url, headers=headers)



print(response.json())