# orchestration_layer/main.py

from dotenv import load_dotenv
load_dotenv()

import os
import httpx
import asyncio
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

# --- FastAPI Initialization ---
app = FastAPI(
    title="Linear Workflow Orchestration Service",
    description="A service that follows a simple, linear workflow to generate music from a natural language query.",
    version="4.0.0" # Version bump for non-agentic architecture
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Service URLs ---
# These are the direct URLs to the other microservices.
MOOD_ANALYSIS_URL = os.getenv("MOOD_ANALYSIS_URL", "http://127.0.0.1:8001/analyze_mood_from_text/")
MUSIC_INITIATE_URL = os.getenv("MUSIC_INITIATE_URL", "http://127.0.0.1:8002/initiate_music_generation/")
MUSIC_STATUS_URL = os.getenv("MUSIC_STATUS_URL", "http://127.0.0.1:8002/get_music_generation_status/")


# --- API Endpoint Models ---
class NaturalLanguageWeatherMusicRequest(BaseModel):
    natural_language_query: str = Field(..., description="Natural language query about weather.")
    duration_seconds: int = Field(90, ge=10, le=300, description="Desired music duration in seconds.")
    # Added optional theme parameter for the caller
    theme: Optional[str] = Field(None, description="Optional music theme to influence generation.")

class MusicResponse(BaseModel):
    music_url: Optional[str] = Field(None, description="URL of the generated music.")
    error: Optional[str] = Field(None, description="Error message if music generation failed.")
    mood: Optional[str] = Field(None, description="Inferred mood from the weather.")
    # Optionally, you might want to return intensity and theme here too,
    # but for now, sticking to the original MusicResponse structure.


# --- Orchestration Endpoint ---
@app.post("/orchestrate/weather-to-music", response_model=MusicResponse)
async def weather_to_music_endpoint(request_data: NaturalLanguageWeatherMusicRequest) -> MusicResponse:
    """
    This endpoint orchestrates the entire workflow without an agent.
    1. Gets the mood and intensity.
    2. Initiates music generation.
    3. Polls for completion.
    """
    mood = None
    intensity = None # Initialize intensity
    theme = request_data.theme # Get theme from request data

    try:
        # Step 1: Get the mood and intensity from the mood analysis service.
        async with httpx.AsyncClient() as client:
            print(f"Calling Mood Analysis Service with query: '{request_data.natural_language_query}'")
            mood_response = await client.post(
                MOOD_ANALYSIS_URL,
                json={"natural_language_query": request_data.natural_language_query},
                timeout=30.0
            )
            mood_response.raise_for_status() # Will raise an exception for 4xx/5xx responses
            mood_data = mood_response.json()
            mood = mood_data.get("mood")
            # Assuming the mood analysis service now returns an 'intensity' field
            intensity = mood_data.get("intensity", 0.6) # Default to 0.6 if not provided by mood service

            if not mood:
                raise HTTPException(status_code=500, detail="Mood Analysis Service did not return a mood.")
            
            print(f"Successfully determined mood: {mood}, Intensity: {intensity}")

            # Step 2: Initiate music generation.
            print(f"Initiating music generation for mood: {mood}, Intensity: {intensity}, Theme: {theme}")
            init_params = {
                "mood": mood,
                "intensity": intensity, # Use dynamic intensity
                "theme": theme,         # Use dynamic/optional theme
                "duration_seconds": request_data.duration_seconds,
                "mock_response": True
            }
            init_response = await client.post(MUSIC_INITIATE_URL, json=init_params, timeout=30.0)
            init_response.raise_for_status()
            init_data = init_response.json()
            task_id = init_data.get("task_id")

            if not task_id:
                raise HTTPException(status_code=500, detail="Music Generation Service did not return a task_id.")
            
            print(f"Music generation task started with ID: {task_id}")

            # Step 3: Poll for the result.
            max_retries = 24 # e.g., 24 retries * 5 seconds = 2 minutes timeout
            poll_interval = 5
            for i in range(max_retries):
                print(f"Polling status for task {task_id} (Attempt {i+1}/{max_retries})")
                await asyncio.sleep(poll_interval)
                
                status_response = await client.post(MUSIC_STATUS_URL, json={"task_id": task_id}, timeout=30.0)
                status_response.raise_for_status()
                status_data = status_response.json()
                
                task_status = status_data.get("status")
                if task_status == "completed":
                    print("Music generation completed.")
                    return MusicResponse(music_url=status_data.get("music_url"), mood=mood)
                elif task_status == "failed":
                    print(f"Music generation failed: {status_data.get('error')}")
                    return MusicResponse(error=status_data.get("error", "Music generation failed."), mood=mood)
                # If status is 'pending' or something else, the loop continues.
            
            return MusicResponse(error="Music generation timed out.", mood=mood)

    except httpx.HTTPStatusError as e:
        # This will catch errors from the downstream services (mood, music)
        error_detail = f"Error calling downstream service at {e.request.url}: {e.response.status_code} {e.response.reason_phrase}"
        try:
            # Try to include the error message from the service if available
            service_error = e.response.json().get("detail")
            if service_error:
                error_detail += f" - {service_error}"
        except Exception:
            pass # Ignore if response body is not valid JSON
        print(f"ERROR: {error_detail}")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=error_detail)
    except Exception as e:
        # Catch any other unexpected errors
        print(f"UNHANDLED ERROR: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unhandled error occurred: {e}")
