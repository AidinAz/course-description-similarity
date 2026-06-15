import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_SYSTEM_PROMPT = """\
You are an academic course analyst. Given two course descriptions, provide a concise analysis with four sections:
1. Shared Topics - content or skills covered by both courses
2. Unique to Course 1 - topics or emphases present only in the first description
3. Unique to Course 2 - topics or emphases present only in the second description
4. Overall Assessment - a brief qualitative judgment of how similar or different the courses are

Be specific and reference the actual subject matter from the descriptions.\
"""


def compare_courses_with_llm(
    first_text: str,
    second_text: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> str:
    resolved_key = api_key or os.environ.get("LLM_API_KEY")
    resolved_url = base_url or os.environ.get("LLM_BASE_URL")
    resolved_model = model or os.environ.get("LLM_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")

    if not resolved_key:
        raise ValueError("No API key found. Set LLM_API_KEY in .env or pass api_key=")
    if not resolved_url:
        raise ValueError("No base URL found. Set LLM_BASE_URL in .env or pass base_url=")

    client = OpenAI(api_key=resolved_key, base_url=resolved_url)

    user_message = (
        f"Course 1:\n{first_text.strip()}\n\n"
        f"Course 2:\n{second_text.strip()}"
    )

    response = client.chat.completions.create(
        model=resolved_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("LLM returned no text content.")
    return content


if __name__ == "__main__":
    course_first_text = """
    First course description...
    """

    course_second_text = """
    Second course description...
    """

    analysis = compare_courses_with_llm(course_first_text, course_second_text)

    print(f"\n{'=' * 50}")
    print("LLM Course Comparison Analysis")
    print('=' * 50)
    print(analysis)
