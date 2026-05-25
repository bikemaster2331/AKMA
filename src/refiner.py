from groq import Groq
from config import GROQ_API_KEY
from prompts import REFINE_DOCUMENT_PROMPT

client = Groq(api_key=GROQ_API_KEY)

def refine_document(user_query: str, document_original: str) -> dict:
    prompt = REFINE_DOCUMENT_PROMPT.format(
        document_original=document_original,
        user_query=user_query
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        temperature=0.3
    )

    text = response.choices[0].message.content.strip()

    mutation_type = "expansion"
    refined_text = ""
    in_refined_section = False

    for line in text.split("\n"):
        if line.startswith("<<<MUTATION_TYPE>>>:"):
            raw_type = line.split(":", 1)[1].strip().lower()
            mutation_type = raw_type if raw_type in ["correction", "expansion"] else "expansion"
        elif line.startswith("<<<REFINED_DOCUMENT>>>:"):
            in_refined_section = True
        elif line.startswith("<<<CHANGES_MADE>>>:"):
            in_refined_section = False
        elif in_refined_section:
            refined_text += line + "\n"

    return {
        "refined_text": refined_text.strip(),
        "mutation_type": mutation_type
    }