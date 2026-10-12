# Refactorización a LCEL asíncrono

Una cadena `prompt | modelo | StrOutputParser` toma `{"contexto", "pregunta"}` y devuelve texto plano.

## Cómo correrlo

Requiere Python 3.12.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e .
cp .env.example .env
```

Completa la clave del proveedor y ejecuta:

```bash
.venv/bin/python lcel_async.py
```

`LLM_PROVIDER` vale `openai` (por defecto) o `anthropic`.

## Tests

No llaman a la API.

```bash
.venv/bin/python -m unittest discover -s tests -v
```
