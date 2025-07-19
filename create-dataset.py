import ollama
from ollama import ChatResponse
import json
import time
import dotenv
import os

from utils.util import calculateDuration

dotenv.load_dotenv()

OLLAMA_MODEL = os.getenv('OLLAMA_MODEL')


def generatePrompt(law_item):
    return f"""
    You are a legal QA generator for Bangladeshi law.

    Given the following law section, generate atleast 2 unique QA pairs that a citizen might ask about this law. The number of pairs will vary depending on content size (larger the content more question pairs) upto 10. Each answer must include the section number and a clear, human-friendly explanation.

    Provide questions and answers will be in English and Bangla language, The ration will be BN:EN=1:3.

    LAW NAME: {law_item.get('law_name_en')}
    PART: {law_item.get('part_no_en')} - ({law_item.get('part_name_en')})
    CHAPTER: {law_item.get('chapter_no_en')} - ({law_item.get('chapter_name_en')})
    SECTION NO: {law_item.get('section_no_en')}
    SECTION NAME: {law_item.get('section_name_en')}
    CONTENT:
    {law_item.get('content')}

    **The output must be in JSON format even if there is only one QA pair**
    Format output as a JSON array of:
    {{
        "output": [
			{{
				"question": "...",
				"answer": "...",
				"law_reference": "Act name, Part, Chapter, Section number", // If anything missing do not set to None or null just spik it.
				"input_context": "Full content"
			}},
			...
    	]
    }}
    """


def generateQA(law_item) -> list[object]:
    response: ChatResponse = ollama.chat(model=OLLAMA_MODEL, stream=False, messages=[
        {
            'role': 'user',
            'content': generatePrompt(law_item),
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


def main():
    start_time = time.time()
    input_files = ['input-data/state-aquisition.json',
                   'input-data/the-transfer-of-property-act.json']
    output_files = ['output-data/state-aquisition.jsonl',
                    'output-data/the-transfer-of-property-act.jsonl']

    for f_i, file in enumerate(input_files):
        print("Processing File, "+file)
        with open(file, 'r') as f:
            law_items = json.load(f)

        dataset = []
        for i, law_item in enumerate(law_items):
            print(
                f"{i+1} - ({law_item.get('section_no_en')}) {law_item.get('section_name_en')} ...", end="")
            qa_pairs = generateQA(law_item)
            # print(json.dumps(qa_pairs, indent=2))
            for qa in qa_pairs.get('output'):
                dataset.append(qa)
            print(f" {len(qa_pairs.get('output'))} pairs")

        with open(output_files[f_i], 'w') as f:
            print(
                f"Saving {len(dataset)} pairs to {output_files[f_i]}...", end="")
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print("Done")

    execution_time = calculateDuration(start_time, time.time())
    print(
        f"Script execution time: {execution_time[0]} hours, {execution_time[1]} minutes, {execution_time[2]} seconds")


if __name__ == "__main__":
    main()