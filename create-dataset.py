import ollama
from ollama import ChatResponse, Client
import json
import time
import dotenv
import os
import random

from utils.util import calculateDuration

dotenv.load_dotenv()

OLLAMA_MODEL = os.getenv('OLLAMA_MODEL')

client = Client()


def generatePrompt(law_items: list[dict]) -> str:
    context = []
    for item in law_items:
        context.append(f"""
        ACT NAME: {item.get('law_name_en') or item.get('law_name_bn')}
        REFERENCE URL: {item.get('ref_url')}
        PART: {item.get('part_no_en') or item.get('part_no_bn','')} ({item.get('part_name_en') or item.get('part_name_bn','')})
        CHAPTER: {item.get('chapter_no_en') or item.get('chapter_no_bn','')} ({item.get('chapter_name_en') or item.get('chapter_name_bn','')})
        SECTION: {item.get('section_no_en') or item.get('section_no_bn','')} {item.get('section_name_en') or item.get('section_name_bn','')}
        CONTENT:
        {item.get('content')}
        """)

    return f"""
    You are a legal QA generator for Bangladeshi law.

    Given the following law section, generate 3 to 50 unique QA pairs that a citizen might ask about this law. The number of pairs will vary depending on content size (larger the content more question pairs) upto 50. Each answer must include the section number and a clear, human-friendly explanation.

    Provide questions and answers will be in English and Bangla language.

    Context:::
    {"\n\n---\n\n".join(context)}

    -----

    Note: **The output must be in JSON format even if there is only one QA pair**
    Format output as a JSON array of:
    {{
        "output": [
			{{
				"question": "...",
				"answer": "...",
				"law_reference": <Act name, Part, Chapter, Section numbers>, // If anything missing do not set to None or null just spik it. If the context has is multiple content then set multiple section, chapter and part no if exists.
                "context": <Summary of the content and reference url> // If reference url exists on the context set it. If not just skip it.
			}},
			...
    	]
    }}

    Make sure, do not provide any extra text or information rather than the JSON.
    """


def generateQA(law_items: list[dict]) -> list[object]:
    response: ChatResponse = client.chat(model=OLLAMA_MODEL, stream=False, think=False, messages=[
        {
            'role': 'user',
            'content': generatePrompt(law_items),
        },
    ], format="json")

    try:
        response_text = response['message']['content'].strip()
        qa_pairs = json.loads(response_text)
        return qa_pairs
    except Exception as e:
        print("Error parsing LLM response:", e)
        print("Response text:", response['message']['content'])
        return []


def saveItems(file, items):
    if len(items) == 0:
        print('Skip')
        return
    try:
        with open(file, '+a', encoding="utf-8") as f:
            for qa in items.get('output'):
                json.dump(qa, f, ensure_ascii=False)
                f.write("\n")
            print(f" {len(items.get('output'))} pairs")
    except Exception as e:
        print("Error while parsing LLM output: ", e, items)


def main():
    start_time = time.time()
    input_files = ['input-data/state-aquisition.json', 'input-data/registration-act.json'
                   'input-data/the-transfer-of-property-act.json', 'others.json']
    output_files = ['output-data/state-aquisition.jsonl', 'output-data/registration-act.jsonl'
                    'output-data/the-transfer-of-property-act.jsonl', 'others.jsonl']

    for f_i, file in enumerate(input_files):
        print("Processing File, "+file)
        with open(file, 'r') as f:
            law_items = json.load(f)

        for i, law_item in enumerate(law_items):
            print(
                f"IDX {i+1} - Section [{law_item.get('section_no_en') or law_item.get('section_no_bn')}] ...", end="")
            qa_pairs = generateQA([law_item])
            saveItems(output_files[f_i], qa_pairs)

            rand_items = random.sample(law_items, i+1)
            print(
                f"\tRandom Sections {[item.get('section_no_en') or item.get('section_no_bn') for item in rand_items]} ...", end="")
            qa_pairs = generateQA(rand_items)
            saveItems(output_files[f_i], qa_pairs)

    execution_time = calculateDuration(start_time, time.time())
    print(
        f"Script execution time: {execution_time[0]} hours, {execution_time[1]} minutes, {execution_time[2]} seconds")


if __name__ == "__main__":
    main()
