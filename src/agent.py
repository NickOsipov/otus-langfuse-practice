"""Module: agent
Description: Агент поддержки магазина: поиск в базе знаний, ответ OpenAI, трасса Langfuse.
"""

import argparse

from langfuse import get_client, observe, propagate_attributes
from langfuse.openai import OpenAI

from config.variables import OPENAI_API_KEY, OPENAI_MODEL

# Конфигурация магазина и политики возврата, доставки и гарантии.
SHOP = "Орбита"
SOURCE_LINE = f"Источник: правила магазина {SHOP}"

POLICIES: dict[tuple[str, ...], str] = {
    ("возврат", "вернуть", "верну"): (
        "Исправный товар можно вернуть в течение 14 дней после получения заказа, "
        "если сохранены упаковка и комплектация. Деньги возвращаются на карту "
        "в течение 10 рабочих дней."
    ),
    ("доставка", "доставить", "довезти"): (
        "Курьерская доставка по городу - 1-2 дня, в другие регионы - 3-7 дней. "
        "Заказы дороже 5000 рублей доставляются бесплатно."
    ),
    ("гарантия", "гарантий"): (
        "Гарантия на электронику - 12 месяцев с даты получения заказа. "
        "Для обращения нужен номер заказа."
    ),
}

PROMPTS: dict[str, str] = {
    "v1": "Ответь клиенту интернет-магазина кратко, используя переданный контекст.",
    "v2": (
        f"Ответь клиенту интернет-магазина {SHOP} только по переданному контексту. "
        "Если данных недостаточно, прямо скажи об этом и предложи обратиться в поддержку. "
        f"Заверши ответ строкой «{SOURCE_LINE}»."
    ),
}

# observe - это декоратор Langfuse, который отслеживает выполнение функции и записывает трассу.
@observe(name="knowledge-search")
def search_knowledge(question: str) -> str:
    """Найти статьи базы знаний по ключевым словам в вопросе клиента."""
    normalized = question.lower()
    matches = [
        text
        for keywords, text in POLICIES.items()
        if any(keyword in normalized for keyword in keywords)
    ]
    return "\n".join(matches) or "Подходящая статья в базе знаний не найдена."


@observe(name="support-agent")
def answer_question(
    question: str,
    prompt_version: str,
    user_id: str,
    session_id: str,
) -> str:
    """Собрать ответ агента, записать трассу Langfuse и оценку source-present."""


    with propagate_attributes(
        user_id=user_id,
        session_id=session_id,
        version=prompt_version,
        tags=["agentops-open-lesson", prompt_version],
        metadata={"provider": "openai", "application": "shop-support-agent"},
    ):
        context = search_knowledge(question)
        client = OpenAI(api_key=OPENAI_API_KEY)
        completion = client.chat.completions.create(
            name="compose-support-answer",
            model=OPENAI_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": PROMPTS[prompt_version]},
                {
                    "role": "user",
                    "content": f"Контекст:\n{context}\n\nВопрос:\n{question}",
                },
            ],
        )
        answer = completion.choices[0].message.content or ""

        get_client().score_current_trace(
            name="source-present",
            value=float(SOURCE_LINE in answer),
            data_type="BOOLEAN",
            comment="Детерминированная проверка требования к формату ответа",
        )
        return answer


def parse_args() -> argparse.Namespace:
    """Разобрать аргументы CLI демо-агента."""
    parser = argparse.ArgumentParser(
        description=f"Демо агента поддержки интернет-магазина {SHOP}"
    )
    parser.add_argument(
        "--question",
        default="Можно вернуть наушники через 10 дней после получения заказа?",
    )
    parser.add_argument("--prompt-version", choices=PROMPTS, default="v1")
    parser.add_argument("--user-id", default="customer-2045")
    parser.add_argument("--session-id", default="order-A-1043")
    return parser.parse_args()


def main() -> None:
    """Ответить на вопрос и отправить трассу в Langfuse."""
    args = parse_args()
    answer = answer_question(
        question=args.question,
        prompt_version=args.prompt_version,
        user_id=args.user_id,
        session_id=args.session_id,
    )
    print(answer)
    get_client().flush()


if __name__ == "__main__":
    main()
