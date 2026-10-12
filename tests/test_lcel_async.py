"""Cubre sintaxis LCEL, asincronía y la configuración del prompt y del modelo."""

import asyncio
import inspect
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import lcel_async


class TestLcelSyntax(unittest.TestCase):
    def test_pipe_composes_prompt_model_and_parser(self) -> None:
        model = FakeListChatModel(responses=["texto"])
        chain = lcel_async.build_chain(model)
        self.assertEqual(
            [type(step).__name__ for step in chain.steps],
            ["ChatPromptTemplate", "FakeListChatModel", "StrOutputParser"],
        )
        self.assertIs(chain.first, lcel_async.prompt)
        self.assertIsInstance(chain.last, StrOutputParser)

    def test_source_uses_the_pipe_operator(self) -> None:
        source = Path(lcel_async.__file__).read_text(encoding="utf-8")
        self.assertIn("return prompt | model | StrOutputParser()", source)


class TestAsyncExecution(unittest.IsolatedAsyncioTestCase):
    async def test_ainvoke_returns_plain_text(self) -> None:
        model = FakeListChatModel(responses=["La entropía mide el desorden."])
        chain = lcel_async.build_chain(model)
        text = await chain.ainvoke(
            {"contexto": "fisica", "pregunta": "¿Qué es la entropía?"}
        )
        self.assertIsInstance(text, str)
        self.assertIn("desorden", text)

    def test_main_is_a_coroutine_that_awaits_ainvoke(self) -> None:
        self.assertTrue(inspect.iscoroutinefunction(lcel_async.main))
        source = Path(lcel_async.__file__).read_text(encoding="utf-8")
        self.assertIn("respuesta = await chain.ainvoke(", source)
        self.assertIn("asyncio.run(main())", source)


class TestPromptAndModel(unittest.TestCase):
    def test_prompt_roles_are_system_and_human(self) -> None:
        self.assertIsInstance(lcel_async.prompt, ChatPromptTemplate)
        self.assertEqual(set(lcel_async.prompt.input_variables), {"contexto", "pregunta"})
        messages = lcel_async.prompt.format_messages(
            contexto="fisica",
            pregunta="¿Qué es la entropía?",
        )
        self.assertEqual([message.type for message in messages], ["system", "human"])

    def test_openai_model_uses_model_name_and_temperature(self) -> None:
        with patch.dict(
            os.environ,
            {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "sk-test-not-real"},
        ):
            model = lcel_async.chat_model()
        self.assertIsInstance(model, lcel_async.ChatOpenAI)
        self.assertEqual(model.model_name, lcel_async.OPENAI_MODEL_NAME)
        self.assertEqual(model.temperature, lcel_async.TEMPERATURE)


if __name__ == "__main__":
    unittest.main()
