import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_SYSTEM_PROMPT = """\
You are an academic course analyst. Compare two course descriptions and produce a structured analysis using exactly these five sections:

### Shared Topics
- List 2-5 specific topics, skills, or learning outcomes covered by both courses.
- One bullet per item. If there is no overlap, write "None identified."

### Unique to Course 1
- List 2-5 specific topics or emphases found only in Course 1.
- One bullet per item. If there is nothing unique, write "None identified."

### Unique to Course 2
- List 2-5 specific topics or emphases found only in Course 2.
- One bullet per item. If there is nothing unique, write "None identified."

### Overall Assessment
Write 2-3 sentences summarizing how similar or different the courses are in purpose, audience, and content.

### Similarity Score: <number>/100
Replace <number> with a single integer from 0 to 100 using this rubric:
- 0-20: Different disciplines or unrelated purposes
- 21-40: Same broad domain, little content overlap
- 41-60: Related field with moderate content overlap
- 61-80: Substantially similar topics and learning outcomes
- 81-100: Nearly identical courses differing only in minor details

Rules:
- Only reference content explicitly stated in the descriptions; do not infer or invent topics.
- Do not include any text outside these five sections.\
"""

def compare_courses_with_llm(
    first_text: str,
    second_text: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int | None = None,
    timeout: float = 60.0,
    max_retries: int = 2,
) -> str:
    if not isinstance(first_text, str) or not isinstance(second_text, str):
        raise TypeError("Both course descriptions must be strings.")
    if not first_text.strip() or not second_text.strip():
        raise ValueError("Both course descriptions must be non-empty.")

    resolved_key = api_key or os.environ.get("LLM_API_KEY")
    resolved_url = base_url or os.environ.get("LLM_BASE_URL")
    resolved_model = model or os.environ.get("LLM_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")

    if not resolved_key:
        raise ValueError("No API key found. Set LLM_API_KEY in .env or pass api_key=")
    if not resolved_url:
        raise ValueError("No base URL found. Set LLM_BASE_URL in .env or pass base_url=")

    client = OpenAI(api_key=resolved_key, base_url=resolved_url, max_retries=max_retries)

    user_message = (
        f"<course_1>\n{first_text.strip()}\n</course_1>\n\n"
        f"<course_2>\n{second_text.strip()}\n</course_2>"
    )

    create_kwargs: dict = dict(
        model=resolved_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=temperature,
        timeout=timeout,
    )
    if max_tokens is not None:
        create_kwargs["max_tokens"] = max_tokens

    response = client.chat.completions.create(**create_kwargs)

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

