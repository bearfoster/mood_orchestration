# Orchestration Layer (Master Agent)

This service acts as the "brain" of the **Weather to Mood to Music** agentic AI system. It manages the overall workflow, sequences agent interactions, and provides the user-facing API for the entire system. It utilizes a Large Language Model (LLM) to parse natural language requests, extract relevant parameters, orchestrate calls to other services (Mood Analysis, Music Generation), and manage asynchronous processes like music generation polling.

---

## Features

- **Natural Language Understanding:** Accepts natural language queries about weather (e.g., "weather in Sydney today," "it's hot and sunny").
- **LLM-Powered Parameter Extraction:** Uses an LLM to infer structured weather parameters (temperature, conditions) from natural language input.
- **Agentic Orchestration:** Employs LangChain's Agent Executor to intelligently select and sequence calls to other MCP-exposed services.
- **Workflow Management:** Handles multi-step processes, including initiating music generation and then polling for its completion status.
- **Unified API:** Provides a single, clean API endpoint for the frontend web application to interact with the entire system.
- **FastAPI Backend:** Built on FastAPI, offering robust and asynchronous API handling.

---

## API Endpoint

The service exposes a single main endpoint for the web application or other clients:

### `POST /orchestrate/weather-to-music`

**Description:**  
Orchestrates the full weather-to-mood-to-music workflow for a given natural language weather query.

**Input Schema (`NaturalLanguageWeatherMusicRequest`):**
```json
{
  "natural_language_query": "weather in New York on April 2 2021",
  "duration_seconds": 90
}
```
- `natural_language_query` (string): A descriptive string about the weather, which the LLM will then parse.
- `duration_seconds` (integer, optional): Desired duration of the music in seconds (default: 90, min: 10, max: 300).

**Output Schema (`MusicResponse`):**
```json
{
  "music_url": "https://mock-music-cdn.com/generated_music/task_id_mood.mp3",
  "error": null,
  "mood": "joyful"
}
```
> Note: `music_url` will be null and `error` will contain a message if music generation fails or times out.

---

## Dependencies

This service relies on:

- **Mood Analysis Service** (running on http://127.0.0.1:8001)
- **Music Generation Service** (running on http://127.0.0.1:8002)
- **Azure OpenAI deployment** for its LLM capabilities.

---

## Setup and Installation

### Prerequisites

- Python 3.12 (recommended)
- pip (Python package installer)
- Git Bash (or another terminal where `source` command works on Windows)
- An Azure OpenAI API key, endpoint, API version, and a deployed chat model.

### Steps

#### 1. Clone the Repository (if applicable) or navigate to the service directory:
```sh
cd path/to/your/project/orchestration_layer
```

#### 2. Create a Virtual Environment
It's highly recommended to use a virtual environment to manage dependencies.
```sh
python -m venv venv_orchestration
```

#### 3. Activate the Virtual Environment

- **Git Bash:**
  ```sh
  source venv_orchestration/Scripts/activate
  ```
- **Command Prompt:**
  ```sh
  venv_orchestration\Scripts\activate
  ```
- **PowerShell:**
  ```sh
  .\venv_orchestration\Scripts\Activate.ps1
  ```

You should see `(venv_orchestration)` at the start of your terminal prompt.

#### 4. Install Dependencies

Ensure you have the `requirements.txt` file in your orchestration_layer directory. Then, with your virtual environment activated:
```sh
pip install -r requirements.txt
```

#### 5. Set Environment Variables for Azure OpenAI

This service requires access to your Azure OpenAI deployment. Set the following environment variables in your terminal before running the service:

- **Git Bash:**
  ```sh
  export AZURE_OPENAI_API_KEY="YOUR_ACTUAL_API_KEY"
  export AZURE_OPENAI_ENDPOINT="https://YOUR_RESOURCE_NAME.openai.azure.com/"
  export AZURE_OPENAI_API_VERSION="2024-02-15-preview"
  export AZURE_OPENAI_CHAT_MODEL_NAME="your-deployment-name"
  ```
- **Command Prompt:**
  ```cmd
  set AZURE_OPENAI_API_KEY=YOUR_ACTUAL_API_KEY
  set AZURE_OPENAI_ENDPOINT=https://YOUR_RESOURCE_NAME.openai.azure.com/
  set AZURE_OPENAI_API_VERSION=2024-02-15-preview
  set AZURE_OPENAI_CHAT_MODEL_NAME=your-deployment-name
  ```
- **PowerShell:**
  ```powershell
  $env:AZURE_OPENAI_API_KEY="YOUR_ACTUAL_API_KEY"
  $env:AZURE_OPENAI_ENDPOINT="https://YOUR_RESOURCE_NAME.openai.azure.com/"
  $env:AZURE_OPENAI_API_VERSION="2024-02-15-preview"
  $env:AZURE_OPENAI_CHAT_MODEL_NAME="your-deployment-name"
  ```

---

## Running the Service

With the virtual environment activated and environment variables set, run the FastAPI application using Uvicorn:

```sh
uvicorn main:app --reload --port 8003
```

The service will start and be accessible at [http://127.0.0.1:8003](http://127.0.0.1:8003). The `--reload` flag ensures that the server automatically reloads if you make changes to the code.

---

## Testing the Service (End-to-End)

To test the full system, ensure the following services are running in separate terminals/processes:

- **Mood Analysis Service:** http://127.0.0.1:8001
- **Music Generation Service:** http://127.0.0.1:8002

Then, you can interact with the Orchestration Layer via its API or, ideally, via the provided Web Application.

### Testing via curl (Direct API Call):

```sh
curl -X POST "http://127.0.0.1:8003/orchestrate/weather-to-music" \
     -H "Content-Type: application/json" \
     -d '{
           "natural_language_query": "What kind of music fits a breezy, sunny afternoon in Sydney?",
           "duration_seconds": 90
         }'
```

**Expected Output (similar to):**
```json
{
  "music_url": "https://mock-music-cdn.com/generated_music/some-task-id_serene.mp3",
  "error": null,
  "mood": "serene"
}
```

### Testing via the Web Application

Open your `index.html` file in a web browser and interact with the orchestration layer through the provided UI.

---