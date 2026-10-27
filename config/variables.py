"""Конфигурация из переменных окружения."""

import os

from dotenv import load_dotenv

load_dotenv()


def _required(name: str) -> str:
    """Вернуть обязательную переменную окружения или упасть с понятной ошибкой."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Не задана переменная окружения {name}")
    return value


OPENAI_API_KEY = _required("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
