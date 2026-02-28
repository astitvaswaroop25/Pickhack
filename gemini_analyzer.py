import os, json
from google import genai
from google.genai.types import GenerateContentConfig, Part
from dotenv import load_dotenv

load_dotenv()
_key = os.getenv("GEMINI_API_KEY")
print(f"DEBUG: Using API Key ending in ...{_key[-4:] if _key else 'NOT SET - add GEMINI_API_KEY to .env'}")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

SYSTEM_PROMPT = """You are a smart city traffic analysis AI. Analyze the image carefully and return JSON:
{
  "vehicles": [{"type": "car|truck|bus", "box_2d": [y_min, x_min, y_max, x_max]}],
  "emergency_vehicles": [{"type": "ambulance|police", "box_2d": [y_min, x_min, y_max, x_max]}],
  "pedestrians": [{"box_2d": [y_min, x_min, y_max, x_max], "crossing": true}],
  "hands": [{"box_2d": [y_min, x_min, y_max, x_max]}],
  "traffic_density": "low|medium|high",
  "recommended_action": "description",
  "emergency_priority": true|false
}
Rules:
- All box_2d coordinates must be normalized to the range 0-1000.
- Draw boxes as tightly as possible around each detected object.
- Only include objects you are highly confident are present. Do not guess.
- For hands: detect any clearly visible human hand or wrist in the frame."""

def analyze_frame(frame_bytes: bytes) -> dict:
    image_part = Part.from_bytes(data=frame_bytes, mime_type="image/jpeg")
    config = GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.3,
        response_mime_type="application/json"
    )
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[image_part, "Analyze this image."],
        config=config
    )
    return json.loads(response.text)
