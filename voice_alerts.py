import os
from elevenlabs.client import ElevenLabs
from elevenlabs import play
from dotenv import load_dotenv

load_dotenv()
el_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

def generate_alert(event_type: str, details: str):
    messages = {
        "emergency": f"Attention: {details}. All vehicles clear the intersection.",
        "pedestrian": f"Pedestrian crossing detected. {details}.",
        "status": f"Traffic status update: {details}."
    }
    text = messages.get(event_type, details)
    audio = el_client.text_to_speech.convert(
        text=text,
        voice_id="pNInz6obpgDQGcFmaJgB", # "Adam"
        model_id="eleven_turbo_v2_5"
    )
    return audio

def play_alert(event_type: str, details: str):
    audio = generate_alert(event_type, details)
    play(audio)