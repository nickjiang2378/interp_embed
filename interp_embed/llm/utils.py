from openai import OpenAI, AsyncOpenAI
import os
from typing import Union
import json
from typing import List, Dict, Optional, Any  # Added missing imports

def get_llm_client(is_openai_model: bool = False, is_async: bool = False) -> Union[OpenAI, AsyncOpenAI]:
    """
    Get the appropriate LLM client based on the model name.

    Args:
        model: Model identifier (e.g., "gpt-4o", "google/gemini-2.5-flash", "openai/o3-mini")
        is_async: Whether to return an async client

    Returns:
        OpenAI or AsyncOpenAI client configured for the appropriate service
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    openrouter_url = "https://openrouter.ai/api/v1"

    if is_openai_model:
        # Use OpenAI directly
        assert openai_api_key is not None, "OPENAI_API_KEY is not set"
        if is_async:
            return AsyncOpenAI(api_key=openai_api_key)
        else:
            return OpenAI(api_key=openai_api_key)
    else:
        # Use OpenRouter for other models
        assert openrouter_api_key is not None, "OPENROUTER_API_KEY is not set"
        if is_async:
            return AsyncOpenAI(
                base_url=openrouter_url,
                api_key=openrouter_api_key
            )
        else:
            return OpenAI(
                base_url=openrouter_url,
                api_key=openrouter_api_key
            )

async def call_async_llm(
    client: AsyncOpenAI,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: Optional[int] = None,
    **kwargs
):
    is_openai_model = model.startswith("openai/")
    if is_openai_model:
        model = model[7:]

    is_reasoning_model = model in ["o3-mini", "o4-mini", "o3", "o3-pro", "o1-preview", "o1", "o3-pro"]
    if is_reasoning_model:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )
    elif is_openai_model:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            **kwargs
        )
    else:
        # Ensure a default temperature if not provided in kwargs for non-OpenAI non-reasoning models
        temperature = kwargs.get("temperature", 0.01)
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **{k: v for k, v in kwargs.items() if k != "temperature"}
        )
    return response

def extract_json_from_response(content: str) -> str:
    """
    Extract JSON from LLM response content, handling markdown code blocks.

    Args:
        content: The raw response content from the LLM

    Returns:
        The extracted JSON string
    """
    if "```json" in content:
        json_start = content.find("```json") + 7
        json_end = content.find("```", json_start)
        json_str = content[json_start:json_end].strip()
    elif "```" in content:
        json_start = content.find("```") + 3
        json_end = content.find("```", json_start)
        json_str = content[json_start:json_end].strip()
    else:
        json_str = content.strip()

    return json_str