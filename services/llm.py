"""LangChain clients for the LiteLLM proxy. Every LLM and embedding call goes through here."""

from typing import cast

from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel

from services.config import get_settings


def chat_model() -> ChatOpenAI:
    """The model for every LLM step except code generation. LiteLLM speaks the OpenAI
    API, so the OpenAI client works for any model it serves (including Claude); only
    base_url changes."""
    settings = get_settings()
    return _client(settings.llm_model, settings.llm_temperature)


def code_model() -> ChatOpenAI:
    """The model for the analyst's code generation: LLM_CODE_MODEL, or the general one."""
    settings = get_settings()
    return _client(settings.llm_code_model or settings.llm_model, settings.llm_code_temperature)


def _client(model: str, temperature: float | None) -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        model=model,
        base_url=settings.openai_base_url,
        api_key=settings.litellm_api_key,
        temperature=temperature,  # None: not sent, the model's default applies
        timeout=60,
        max_retries=2,  # retries transient HTTP errors (rate limits, 5xx)
    )


def embeddings() -> OpenAIEmbeddings:
    settings = get_settings()
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        base_url=settings.openai_base_url,
        api_key=settings.litellm_api_key,
        # The proxy isn't OpenAI: send plain strings, not OpenAI-tokenized input
        check_embedding_ctx_length=False,
    )


def with_schema[T: BaseModel](model: BaseChatModel, schema: type[T]) -> Runnable[LanguageModelInput, T]:
    """The model, set up to answer with an instance of schema.

    with_structured_output sends the Pydantic model's JSON schema with each request and
    parses the reply into an instance. Its declared return type doesn't say which
    model comes back, so this wrapper does. [T: BaseModel] declares a type parameter:
    with_schema(model, Selection) returns a Runnable that produces a Selection.
    """
    # LangChain annotates one of the method's parameters as a bare `list`, which strict
    # mode reports as partially unknown.
    structured = model.with_structured_output(schema)  # pyright: ignore[reportUnknownMemberType]
    return cast(Runnable[LanguageModelInput, T], structured)
