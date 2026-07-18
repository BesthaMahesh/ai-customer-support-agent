import os
from openai import OpenAI
from typing import Generator
import config
from utils import get_logger

logger = get_logger("Chatbot")

def get_client() -> OpenAI:
    """Initializes and returns the OpenAI client configured for OpenRouter."""
    if not config.OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY is not set in environment or .env file.")
    return OpenAI(
        api_key=config.OPENROUTER_API_KEY,
        base_url=config.OPENROUTER_API_URL
    )

def get_chat_response(system_prompt: str, user_prompt: str, temperature: float = 0.3) -> str:
    """Sends a chat request to OpenRouter and returns the full response string."""
    try:
        client = get_client()
        logger.info(f"Sending LLM request using model: '{config.OPENROUTER_MODEL}'")
        
        response = client.chat.completions.create(
            model=config.OPENROUTER_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            extra_headers={
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "AI RAG Assistant SaaS",
            }
        )
        if response.choices and len(response.choices) > 0:
            return response.choices[0].message.content
        return "Error: Empty response received from model."
    except Exception as e:
        logger.error(f"Error calling LLM (blocking): {e}")
        return f"Error calling Chatbot API: {str(e)}"

def get_chat_response_stream(system_prompt: str, user_prompt: str, temperature: float = 0.3) -> Generator[str, None, None]:
    """Sends a chat request to OpenRouter and yields tokens as they arrive (streaming)."""
    try:
        client = get_client()
        logger.info(f"Starting LLM streaming request using model: '{config.OPENROUTER_MODEL}'")
        
        response = client.chat.completions.create(
            model=config.OPENROUTER_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            stream=True,
            extra_headers={
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "AI RAG Assistant SaaS (Streaming)",
            }
        )
        
        for chunk in response:
            if chunk.choices and len(chunk.choices) > 0:
                delta_content = chunk.choices[0].delta.content
                if delta_content:
                    yield delta_content
    except Exception as e:
        logger.error(f"Error calling LLM (streaming): {e}")
        yield f"\n[Error calling Chatbot API Stream: {str(e)}]"
