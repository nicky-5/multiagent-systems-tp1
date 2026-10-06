"""LangChain clients for the LiteLLM proxy. Every LLM and embedding call goes through here."""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from services.config import get_settings


def chat_model(temperature: float = 0) -> ChatOpenAI:
    """A chat model behind the proxy. LiteLLM speaks the OpenAI API, so the OpenAI
    client works for any model it serves (including Claude), only base_url changes."""
    settings = get_settings()
    return ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.openai_base_url,
        api_key=settings.litellm_api_key,
        temperature=temperature,
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
