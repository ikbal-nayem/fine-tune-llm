import ollama
from ollama import ChatResponse
import json
import time
import dotenv
import os, random

from utils.util import calculateDuration

dotenv.load_dotenv()

OLLAMA_MODEL = os.getenv('OLLAMA_MODEL')


def generatePrompt(law_items: list[dict]) -> str:
    context = []
    for item in law_items:
        context.append(f"""
        ACT NAME: {item.get('law_name_en') or item.get('law_name_bn')}
        REFERENCE URL: {item.get('ref_url')}
        PART: {item.get('part_no_en') or item.get('part_no_bn')} - ({item.get('part_name_en') or item.get('part_name_bn')})
        CHAPTER: {item.get('chapter_no_en') or item.get('chapter_no_bn')} - ({item.get('chapter_name_en')} or {item.get('chapter_name_bn')})
        SECTION: {item.get('section_no_en') or item.get('section_no_bn')} - {item.get('section_name_en') or item.get('section_name_bn')}
        CONTENT:
        {item.get('content')}
        """)
        
    return f"""
    You are a legal QA generator for Bangladeshi law.

    Given the following law section, generate atleast 3 unique QA pairs that a citizen might ask about this law. The number of pairs will vary depending on content size (larger the content more question pairs) upto 50. Each answer must include the section number and a clear, human-friendly explanation.

    Provide questions and answers will be in English and Bangla language, The ratio will be BN:EN=1:3.

    Context:::
    {"\n\n---\n\n".join(context)}

    -----

    **The output must be in JSON format even if there is only one QA pair**
    Format output as a JSON array of:
    {{
        "output": [
			{{
				"question": "...",
				"answer": "...",
				"law_reference": "Act name, Part, Chapter, Section numbers", // If anything missing do not set to None or null just spik it. If the context has is multiple content then set multiple section, chapter and part no if exists.
                "context": "Summary of content with reference url" // If reference url exists on the context set it. If not just skip it.
			}},
			...
    	]
    }}
    """


def generateQA(law_items: list[dict]) -> list[object]:
    response: ChatResponse = ollama.chat(model=OLLAMA_MODEL, stream=False, messages=[
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
        print("Response text:", response_text)
        return []

def saveItems(file, items):
    try:
        with open(file, '+a', encoding="utf-8") as f:
            for qa in items.get('output'):
                json.dump(qa, f, ensure_ascii=False)
                f.write("\n")
            print(f" {len(items.get('output'))} pairs")
    except Exception as e:
        print("Error while parsing LLM output: ", e)

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
        
        # try:
        #     with open(output_files[f_i], 'r') as f:
        #         existing_data = json.load(f)
        #         print(f"{len(existing_data)} data found")
        # except FileNotFoundError:
        #     existing_data = []

        # dataset = []
        for i, law_item in enumerate(law_items):
            print(f"IDX {i+1} - Section [{law_item.get('section_no_en') or law_item.get('section_no_bn')}] ...", end="")
            qa_pairs = generateQA([law_item])
            saveItems(output_files[f_i], qa_pairs)

            rand_items = random.sample(law_items, i+1)
            print(f"Random Sections {[item.get('section_no_en') or item.get('section_no_bn') for item in rand_items]} ...", end="")
            qa_pairs = generateQA(rand_items)
            saveItems(output_files[f_i], qa_pairs)

        # with open(output_files[f_i], 'w') as f:
        #     print(
        #         f"Saving {len(dataset)} pairs to {output_files[f_i]}...", end="")
        #     f.write(json.dumps(dataset, ensure_ascii=False) + '\n')
        #     print("Done")

    execution_time = calculateDuration(start_time, time.time())
    print(
        f"Script execution time: {execution_time[0]} hours, {execution_time[1]} minutes, {execution_time[2]} seconds")


if __name__ == "__main__":
    main()
