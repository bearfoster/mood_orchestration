# orchestration_layer/main.py

from dotenv import load_dotenv # ADDED
load_dotenv() # ADDED: Load environment variables from .env file

import os
import requests
import json
import time
import asyncio

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Literal, Optional, Union

# LangChain imports
from langchain_core.tools import Tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import JsonOutputParser


# --- FastAPI Application Initialization ---
app = FastAPI(
    title="Orchestration Layer API",
    description="API for orchestrating weather-to-mood-to-music workflow with natural language weather input.",
    version="0.4.0"
)

# --- Azure OpenAI LLM Initialization ---
try:
    llm = AzureChatOpenAI(
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
        azure_deployment=os.getenv("AZURE_OPENAI_CHAT_MODEL_NAME", "gpt-4"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0.0,
    )
    weather_extractor_llm = llm
except Exception as e:
    print(f"Error initializing Azure OpenAI LLM in Orchestration Layer: {e}")
    print("Please ensure AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION, and AZURE_OPENAI_CHAT_MODEL_NAME are set as environment variables.")
    llm = None
    weather_extractor_llm = None


# --- MCPClient for interacting with services ---

class MCPClient:
    def __init__(self, mood_analysis_url: str, music_generation_url: str):
        self.mood_analysis_url = mood_analysis_url
        self.music_generation_url = music_generation_url
        self.tools_cache: Dict[str, Dict[str, Any]] = {}

    async def _discover_tools(self, service_url: str) -> Dict[str, Any]:
        discovery_url = f"{service_url}/mcp/tools"
        try:
            response = requests.get(discovery_url)
            response.raise_for_status()
            tools_data = response.json()
            for tool_name, tool_info in tools_data.items():
                self.tools_cache[tool_name] = tool_info
            print(f"Discovered tools from {service_url}: {list(tools_data.keys())}")
            return tools_data
        except requests.exceptions.RequestException as e:
            print(f"Error discovering tools from {service_url}: {e}")
            return {}

    async def discover_all_tools(self):
        await self._discover_tools(self.mood_analysis_url)
        await self._discover_tools(self.music_generation_url)

    async def invoke_tool(self, tool_name: str, **kwargs: Any) -> Dict[str, Any]:
        tool_info = self.tools_cache.get(tool_name)
        if not tool_info:
            await self.discover_all_tools()
            tool_info = self.tools_cache.get(tool_name)
            if not tool_info:
                 raise ValueError(f"Tool '{tool_name}' not found after discovery. Is the service running?")

        service_id = tool_info.get("service_id")
        if service_id == "mood-analysis-agent-1":
            service_url = self.mood_analysis_url
        elif service_id == "music-generation-agent-2":
            service_url = self.music_generation_url
        else:
            raise ValueError(f"Unknown service_id '{service_id}' for tool '{tool_name}'")

        invocation_url = f"{service_url}/mcp/tool/{tool_name}"
        headers = {"Content-Type": "application/json"}

        print(f"Invoking tool: {tool_name} with args: {kwargs} at {invocation_url}")
        try:
            response = requests.post(invocation_url, headers=headers, json=kwargs)
            response.raise_for_status()
            result = response.json()
            print(f"Tool '{tool_name}' invocation successful. Result: {result}")
            return result
        except requests.exceptions.RequestException as e:
            print(f"Error invoking tool '{tool_name}': {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error invoking tool '{tool_name}': {e}")


mcp_client = MCPClient(
    mood_analysis_url="http://127.0.0.1:8001",
    music_generation_url="http://127.0.0.1:8002"
)


# --- Define New Tool: extract_weather_params ---

class ExtractedWeatherParams(BaseModel):
    temperature_celsius: float = Field(..., description="The inferred temperature in Celsius.")
    conditions: str = Field(..., description="A concise descriptive string of the inferred weather conditions.")

class ExtractWeatherParamsInput(BaseModel):
    """Input for the extract_weather_params tool."""
    natural_language_query: str = Field(..., description="The user's natural language query about the weather.")

async def _extract_weather_params_tool(natural_language_query: str) -> ExtractedWeatherParams:
    """
    Uses an LLM to extract temperature and conditions from a natural language query about weather.
    """
    if weather_extractor_llm is None:
        raise HTTPException(status_code=500, detail="Weather extractor LLM not initialized.")

    weather_extraction_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an AI assistant tasked with extracting current or historical weather parameters "
                "from a user's natural language query. You must identify the temperature in Celsius and "
                "a concise description of the weather conditions. "
                "If temperature or conditions are not explicitly mentioned but implied, try to infer reasonable defaults "
                "or state 'unknown' if no reasonable inference can be made. "
                "Your response must be ONLY a JSON object with 'temperature_celsius' (float) and 'conditions' (string) keys. "
                "Example: {{\"temperature_celsius\": 22.5, \"conditions\": \"sunny with a light breeze\"}}"
            ),
            ("human", "{query}")
        ]
    )

    weather_extraction_chain = weather_extraction_prompt | weather_extractor_llm | JsonOutputParser()

    try:
        llm_response = await weather_extraction_chain.invoke({"query": natural_language_query})
        
        extracted_params = ExtractedWeatherParams(
            temperature_celsius=llm_response.get("temperature_celsius"),
            conditions=llm_response.get("conditions")
        )
        print(f"Extracted weather params for '{natural_language_query}': {extracted_params.model_dump_json()}")
        return extracted_params
    except Exception as e:
        print(f"Error during LLM weather parameter extraction: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to extract weather parameters: {e}. Ensure LLM is configured and responds with valid JSON.")


# --- Define LangChain Tools ---

def create_langchain_tool_from_mcp(mcp_tool_name: str, mcp_client_instance: MCPClient) -> Tool:
    tool_info = mcp_client_instance.tools_cache.get(mcp_tool_name)
    if not tool_info:
        raise ValueError(f"MCP tool '{mcp_tool_name}' not found in client cache for LangChain tool creation.")

    input_schema_dict = tool_info["input_schema"]
    fields = {}
    for k, v in input_schema_dict["properties"].items():
        if "enum" in v and v["type"] == "STRING":
            fields[k] = (Literal[tuple(v["enum"])], Field(..., description=v.get("description", "")))
        elif v["type"] == "NUMBER":
            fields[k] = (float, Field(..., description=v.get("description", "")))
        elif v["type"] == "INTEGER":
            fields[k] = (int, Field(..., description=v.get("description", "")))
        else:
            fields[k] = (str, Field(..., description=v.get("description", "")))

    DynamicInputModel = type(f"{mcp_tool_name.capitalize()}Input", (BaseModel,), fields)

    async def _tool_func(**kwargs: Any) -> Dict[str, Any]:
        return await mcp_client_instance.invoke_tool(mcp_tool_name, **kwargs)

    return Tool(
        name=mcp_tool_name,
        description=tool_info["description"],
        func=_tool_func,
        args_schema=DynamicInputModel
    )


# --- LangChain Agent Executor Setup ---

agent_executor_instance: Optional[AgentExecutor] = None

@app.on_event("startup")
async def startup_event():
    global agent_executor_instance
    if llm is None:
        print("LLM not initialized. Cannot create agent executor.")
        return

    print("Orchestration Layer: Discovering tools and creating agent executor...")
    
    await mcp_client.discover_all_tools()
    
    langchain_tools = []
    langchain_tools.append(
        Tool(
            name="extract_weather_params",
            description="Extracts temperature and conditions from a natural language weather query. "
                        "Input: natural_language_query (string). "
                        "Output: JSON with temperature_celsius (float) and conditions (string).",
            func=_extract_weather_params_tool,
            args_schema=ExtractWeatherParamsInput
        )
    )

    for tool_name in mcp_client.tools_cache.keys():
        try:
            langchain_tools.append(create_langchain_tool_from_mcp(tool_name, mcp_client))
        except Exception as e:
            print(f"Error creating LangChain tool '{tool_name}': {e}. Skipping this tool.")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an AI assistant that orchestrates weather-to-mood-to-music. "
                "Your primary goal is to provide a music URL based on user's weather inquiry. "
                "You have access to the following tools:\n\n"
                "{tools}\n\n"
                "First, use 'extract_weather_params' to get structured temperature and conditions from the user's natural language input. "
                "Then, use 'analyze_weather_mood' with the extracted weather data to determine the mood. "
                "Next, use 'initiate_music_generation' with the inferred mood and a suitable duration (e.g., 90 seconds) to start music creation. "
                "If 'initiate_music_generation' returns a task_id, periodically check its status using 'get_music_generation_status' "
                "until the music_url is available. "
                "Always provide the final music URL to the user in a JSON format: {{'music_url': 'YOUR_URL', 'mood': 'INFERRED_MOOD'}}. "
                "If an error occurs or music cannot be generated, provide an error message in JSON format: {{'error': 'YOUR_ERROR_MESSAGE'}}."
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, langchain_tools, prompt)

    agent_executor_instance = AgentExecutor(agent=agent, tools=langchain_tools, verbose=True)
    print("Orchestration Layer: Agent Executor initialized.")


# --- API Endpoint for the Web App ---

class NaturalLanguageWeatherMusicRequest(BaseModel):
    natural_language_query: str = Field(..., description="Natural language query about weather (e.g., 'weather in Sydney today').")
    duration_seconds: int = Field(90, ge=10, le=300, description="Desired music duration in seconds.")

class MusicResponse(BaseModel):
    music_url: Optional[str] = Field(None, description="URL of the generated music.")
    error: Optional[str] = Field(None, description="Error message if music generation failed.")
    mood: Optional[str] = Field(None, description="Inferred mood from the weather.")

@app.post("/orchestrate/weather-to-music", response_model=MusicResponse)
async def weather_to_music_endpoint(request_data: NaturalLanguageWeatherMusicRequest) -> MusicResponse:
    if agent_executor_instance is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Orchestration service not ready. LLM or Agent Executor not initialized.")
 
    # The agent is designed to handle the full orchestration from a single input.
    # The detailed system prompt guides it through the required steps:
    # 1. Extract weather
    # 2. Analyze mood
    # 3. Initiate music generation
    # 4. Poll for status
    # This simplifies the endpoint logic significantly.
    
    # We combine the user's query and the desired duration into a single input for the agent.
    full_input = (
        f"User weather query: '{request_data.natural_language_query}'. "
        f"Desired music duration: {request_data.duration_seconds} seconds."
    )
    
    print(f"\nAPI Call: Invoking agent for full orchestration with input: '{full_input}'")
    
    try:
        # A single invocation triggers the entire agent-driven workflow.
        response = await agent_executor_instance.invoke(
            {"input": full_input, "chat_history": []}
        )
        print(f"Agent final output: {response.get('output')}")
        # The agent's final output should be the JSON string we requested in the prompt.
        final_data = json.loads(response.get('output'))
        return MusicResponse(**final_data)
    except Exception as e:
        error_message = f"An unhandled error occurred during agent orchestration: {e}"
        print(error_message)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_message)
