"""La cadena LCEL devuelve texto plano. No llama a la API."""

import asyncio
import unittest
from pathlib import Path

from langchain_core.language_models.fake_chat_models import FakeListChatModel

import lcel_async


class TestLcelAsync(unittest.TestCase):
    def test_prompt_has_system_human_and_variables(self) -> None:
        self.assertEqual(set(lcel_async.prompt.input_variables), {"contexto", "pregunta"})
        messages = lcel_async.prompt.format_messages(
            contexto="fisica",
            pregunta="¿Qué es la entropía?",
        )
        self.assertEqual(messages[0].type, "system")
        self.assertEqual(messages[1].type, "human")
        self.assertIn("fisica", messages[0].content)
        self.assertIn("entropía", messages[1].content)

    def test_ainvoke_returns_plain_text(self) -> None:
        model = FakeListChatModel(responses=["La entropía mide el desorden."])
        chain = lcel_async.prompt | model | lcel_async.StrOutputParser()
        text = asyncio.run(
            chain.ainvoke({"contexto": "fisica", "pregunta": "¿Qué es la entropía?"})
        )
        self.assertIsInstance(text, str)
        self.assertIn("desorden", text)

    def test_main_source_uses_lcel_and_await(self) -> None:
        source = Path(lcel_async.__file__).read_text(encoding="utf-8")
        self.assertIn("prompt | model | StrOutputParser()", source)
        self.assertIn("await chain.ainvoke(", source)
        self.assertIn("load_dotenv()", source)


if __name__ == "__main__":
    unittest.main()
