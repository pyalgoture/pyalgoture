from typing import Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI

json_parser = JsonOutputParser()


def get_model(
    model_name: str, model_provider: str, api_key: str, base_url: str | None = None
) -> ChatOpenAI | ChatDeepSeek | None:
    if model_provider == "OpenAI":
        # Get and validate API key
        if not api_key:
            # Print error to console
            print("API Key Error: Please make sure OPENAI_API_KEY is set in your .env file.")
            raise ValueError("OpenAI API key not found.  Please make sure OPENAI_API_KEY is set in your .env file.")
        return ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url)
    elif model_provider == "DeepSeek":
        if not api_key:
            print("API Key Error: Please make sure DEEPSEEK_API_KEY is set in your .env file.")
            raise ValueError("DeepSeek API key not found.  Please make sure DEEPSEEK_API_KEY is set in your .env file.")
        return ChatDeepSeek(model=model_name, api_key=api_key)

    return None


def call_llm(
    prompt: Any,
    model_name: str,
    model_provider: str,
    api_key: str,
    data: dict = {},
    base_url: str | None = None,
    max_retries: int = 3,
    return_json: bool = True,
) -> dict:
    """
    Makes an LLM call with retry logic, handling both JSON supported and non-JSON supported models.

    Args:
        prompt: The prompt to send to the LLM
        model_name: Name of the model to use
        model_provider: Provider of the model
        pydantic_model: The Pydantic model class to structure the output
        max_retries: Maximum number of retries (default: 3)
        default_factory: Optional factory function to create default response on failure

    Returns:
        An instance of the specified Pydantic model
    """

    llm = get_model(model_name, model_provider, api_key, base_url)

    if return_json:
        chain = prompt | llm | json_parser if data else llm | json_parser
    else:
        chain = prompt | llm if data else llm

    # Call the LLM with retries
    for attempt in range(max_retries):
        try:
            # Call the LLM
            result = chain.invoke(data if data else prompt)
            return result  # type: ignore[no-any-return]

        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Error in LLM call after {max_retries} attempts: {e}")
                return {}

    return {}


'''
from typing import Any, TypeVar
import json
from pydantic import BaseModel
T = TypeVar("T", bound=BaseModel)


openai_llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini",  # or "gpt-4" if you have access
    timeout=30,           # timeout in seconds (set to your desired limit)
    max_retries=3
)
deepseek_llm = ChatDeepSeek(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    model="deepseek-chat",
    timeout=30,           # timeout in seconds (set to your desired limit)
    max_retries=3
)

def call_llm(
    prompt: Any,
    model_name: str,
    model_provider: str,
    pydantic_model: type[T],
    api_key: str,
    base_url: str = None,
    max_retries: int = 3,
    default_factory=None,
) -> T:
    """
    Makes an LLM call with retry logic, handling both JSON supported and non-JSON supported models.

    Args:
        prompt: The prompt to send to the LLM
        model_name: Name of the model to use
        model_provider: Provider of the model
        pydantic_model: The Pydantic model class to structure the output
        max_retries: Maximum number of retries (default: 3)
        default_factory: Optional factory function to create default response on failure

    Returns:
        An instance of the specified Pydantic model
    """

    llm = get_model(model_name, model_provider, api_key, base_url)

    # For non-JSON support models, we can use structured output
    if model_provider in [
        "OpenAI",
    ]:
        llm = llm.with_structured_output(
            pydantic_model,
            method="json_mode",
        )

    # Call the LLM with retries
    for attempt in range(max_retries):
        try:
            # Call the LLM
            result = llm.invoke(prompt)

            # For non-JSON support models, we need to extract and parse the JSON manually
            if model_provider not in ["OpenAI"]:
                parsed_result = extract_json_from_response(result.content)
                if parsed_result:
                    return pydantic_model(**parsed_result)
            else:
                return result

        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Error in LLM call after {max_retries} attempts: {e}")
                # Use default_factory if provided, otherwise create a basic default
                if default_factory:
                    return default_factory()
                return create_default_response(pydantic_model)

    # This should never be reached due to the retry logic above
    return create_default_response(pydantic_model)


def create_default_response(model_class: type[T]) -> T:
    """Creates a safe default response based on the model's fields."""
    default_values = {}
    for field_name, field in model_class.model_fields.items():
        if isinstance(field.annotation, str):
            default_values[field_name] = "Error in analysis, using default"
        elif isinstance(field.annotation, float):
            default_values[field_name] = 0.0
        elif isinstance(field.annotation, int):
            default_values[field_name] = 0
        elif hasattr(field.annotation, "__origin__") and isinstance(field.annotation.__origin__, dict):
            default_values[field_name] = {}
        else:
            # For other types (like Literal), try to use the first allowed value
            if hasattr(field.annotation, "__args__"):
                default_values[field_name] = field.annotation.__args__[0]
            else:
                default_values[field_name] = None

    return model_class(**default_values)


def extract_json_from_response(content: str) -> dict | None:
    """Extracts JSON from markdown-formatted response."""
    try:
        json_start = content.find("```json")
        if json_start != -1:
            json_text = content[json_start + 7 :]  # Skip past ```json
            json_end = json_text.find("```")
            if json_end != -1:
                json_text = json_text[:json_end].strip()
                return json.loads(json_text)
    except Exception as e:
        print(f"Error extracting JSON from response: {e}")
    return None

'''
