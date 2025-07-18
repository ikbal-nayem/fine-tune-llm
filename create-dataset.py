import ollama
from ollama import ChatResponse
import json

OLLAMA_MODEL = "gemma3:4b"

response: ChatResponse = ollama.chat(model=OLLAMA_MODEL, messages=[
    {
        'role': 'user',
        'content': 'What do you know about Bangladesh law? tell me in 2 sentences.',
    },
])

print(response['message']['content'])


def generatePrompt(law_item, language='en'):
    return f"""
    You are a legal QA generator for Bangladeshi law.

    Given the following law section, generate 3 unique QA pairs that a citizen might ask about this law. Each answer must include the section number and a clear, human-friendly explanation.

    Provide questions and answers in {'Bangla' if language == 'bn' else 'English'}.

    LAW NAME: {law_item['law_name_en']}
    PART: {law_item['part_no_en']}
    CHAPTER: {law_item['chapter_no_en']} - {law_item['chapter_name_en']}
    SECTION NO: {law_item['section_no_en']}
    SECTION NAME: {law_item['section_name_en']}
    CONTENT:
    {law_item['content']}

    Format output as a JSON array of:
    [
        {{
            "question": "...",
            "answer": "...",
            "law_reference": "Act name, Part, Chapter, Section number",
            "input_context": "Full section content"
        }},
        ...
    ]
    """


def generateQA(law_item, language='en'):
    response: ChatResponse = ollama.chat(model=OLLAMA_MODEL, stream=False, messages=[
        {
            'role': 'user',
            'content': generatePrompt(law_item, language),
        },
    ], format="json")

    try:
        response_text = response['message']['content'].strip()
        qa_pairs = json.loads(response_text)
        return qa_pairs
    except Exception as e:
        print("Error parsing LLM response:", e)
        print("Response text:", response_text)
        return []
