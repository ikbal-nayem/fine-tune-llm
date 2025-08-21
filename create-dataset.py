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
    You are a Legal QA Generator specializing in Bangladeshi law.
    Your task is to generate 3 to 25 unique Question-Answer (QA) pairs from the given law section(s). The number of pairs must depend on the content size (larger content → more pairs, max 25).

    Instructions for QA Pairs:
    - Questions:
        - Should sound like something a general citizen might ask in real life.
        - Do NOT always require section numbers in the question. Example:
            - "What happens if I don’t register my land agreement?"
            - "Can the government cancel my tenancy rights?"
            - "What is Section 23 of The Registration Act?" (only sometimes include law references)
        - Include at least 2 natural Bangla questions.

    - Answers:
        - Must clearly explain the law in simple, human-friendly language.
        - Must always include the relevant Section number(s) and Act name.
        - Should explain how the law applies in practical terms (rights, duties, penalties, or protections).
    
    - Output format must be JSON:
    
    ## Context:::
    {"\n\n---\n\n".join(context)}

    -----

    Note: **The output must be in JSON format for QA pair.** Make sure the JSON
    Format output as a JSON array of:
    {{
        "output": [
			{{
				"question": "...",
				"answer": "...",
				"law_reference": <Act name, Part, Chapter, Section numbers>,
                "context": <Summary of the content and reference url>
			}},
			...
    	]
    }}

    Output Property details,
        - "question": In case of this type question "What is the purpose of Section 2A?" mention law name to the question like this "What is the purpose of Section 2A? of The State Acquisition and Tenancy Act?".
        - "law_reference": If anything missing do not set to None or null just spik it. If the context has is multiple content then set multiple section, chapter and part no if exists.
        - "context": If reference url exists on the context set it. If not just skip it.

    Types of questions to generate:
        - Direct: "What is Section 23 of The Registration Act?"
        - Practical: "What happens if I forget to register my property?"
        - Citizen-centric: "How does <Act name, section name> protect ordinary land buyers?"
        - Bangla: "যদি জমির দলিল রেজিস্ট্রি না করি, তাহলে কী হবে?"
        - Scenario-based: "If two people claim ownership but only one has registration, who will be recognized by law?"
    """


def generateQA(law_items: list[dict]) -> list[object]:
    response: ChatResponse = client.chat(model=OLLAMA_MODEL, stream=False, think=False, messages=[
        {
            'role': 'user',
            'content': generatePrompt(law_items),
        },
    ], format="json", options={"num_predict": 20000})

    try:
        response_text:str = response['message']['content'].strip()
        response_text += "}" if response_text[-1] != "}" else ""
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
    input_files = ['input-data/others.json']
    output_files = ['output-data/others.jsonl']
    # input_files = ['input-data/registration-act.json',
    #                'input-data/the-transfer-of-property-act.json', 'input-data/others.json']
    # output_files = ['output-data/registration-act.jsonl',
    #                 'output-data/the-transfer-of-property-act.jsonl', 'output-data/others.jsonl']

    for f_i, file in enumerate(input_files):
        print("Processing File, "+file)
        with open(file, 'r') as f:
            law_items = json.load(f)

        for i, law_item in enumerate(law_items):
            print(
                f"IDX {i+1} - Section [{law_item.get('section_no_en') or law_item.get('section_no_bn')}] ...", end="", flush=True)
            qa_pairs = generateQA([law_item])
            saveItems(output_files[f_i], qa_pairs)

            rand_items = random.sample(law_items, i+1)
            print(
                f"\tRandom Sections {[item.get('section_no_en') or item.get('section_no_bn') for item in rand_items]} ...", end="", flush=True)
            qa_pairs = generateQA(rand_items)
            saveItems(output_files[f_i], qa_pairs)

    execution_time = calculateDuration(start_time, time.time())
    print(
        f"Script execution time: {execution_time[0]} hours, {execution_time[1]} minutes, {execution_time[2]} seconds")


if __name__ == "__main__":
    main()
