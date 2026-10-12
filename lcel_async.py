import asyncio
import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

TEMPERATURE = 0.7
OPENAI_MODEL_NAME = "gpt-4o-mini"
ANTHROPIC_MODEL_NAME = "claude-3-5-haiku-latest"

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Respondes en español, claro y breve. "
            "Ten en cuenta este contexto cuando aplique: {contexto}",
        ),
        ("human", "{pregunta}"),
    ]
)


def chat_model() -> ChatOpenAI | ChatAnthropic:
    """Modelo de LangChain según LLM_PROVIDER. Por defecto, OpenAI."""
    provider = os.environ.get("LLM_PROVIDER", "openai").strip().lower() or "openai"
    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY", "").strip():
            raise RuntimeError("Falta OPENAI_API_KEY para el proveedor openai.")
        return ChatOpenAI(model_name=OPENAI_MODEL_NAME, temperature=TEMPERATURE)
    if provider == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
            raise RuntimeError("Falta ANTHROPIC_API_KEY para el proveedor anthropic.")
        return ChatAnthropic(model_name=ANTHROPIC_MODEL_NAME, temperature=TEMPERATURE)
    raise RuntimeError(f"LLM_PROVIDER no reconocido: {provider}")


async def main() -> None:
    load_dotenv()
    model = chat_model()
    chain = prompt | model | StrOutputParser()
    respuesta = await chain.ainvoke(
        {
            "contexto": "El usuario está practicando una cadena LCEL asíncrona.",
            "pregunta": "¿Qué es la entropía?",
        }
    )
    print(respuesta)


if __name__ == "__main__":
    asyncio.run(main())
