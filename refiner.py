from groq import Groq
from config import GROQ_API_KEY

client = Groq(api_key=GROQ_API_KEY)

def refine_document(user_query: str, document_original: str) -> dict:
    prompt = f"""You are a knowledge refiner. Your job is to update an existing knowledge document based on new user input.

EXISTING DOCUMENT:
{document_original}

USER QUERY / NEW INFORMATION:
{user_query}

YOUR RULES:
1. Preserve ALL facts from the original document — do not remove or contradict anything
2. Integrate the user's new context, nuance, or information cleanly
3. At the end, briefly state what changed and why
4. Tag the mutation type:
   - 'correction' if you are fixing something wrong in the original
   - 'expansion' if you are adding new information to the original

Respond in EXACTLY this format, with no extra commentary:

MUTATION_TYPE: correction|expansion
REFINED_DOCUMENT:
<your refined document here>
CHANGES_MADE:
<one or two sentences describing what changed>"""

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
        if line.startswith("MUTATION_TYPE:"):
            raw_type = line.split(":", 1)[1].strip().lower()
            mutation_type = raw_type if raw_type in ["correction", "expansion"] else "expansion"
        elif line.startswith("REFINED_DOCUMENT:"):
            in_refined_section = True
        elif line.startswith("CHANGES_MADE:"):
            in_refined_section = False
        elif in_refined_section:
            refined_text += line + "\n"

    return {
        "refined_text": refined_text.strip(),
        "mutation_type": mutation_type
    }