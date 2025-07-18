import ollama
from ollama import ChatResponse
import json

OLLAMA_MODEL = "gemma3:4b"

def generatePrompt(law_item, language='en'):
    return f"""
    You are a legal QA generator for Bangladeshi law.

    Given the following law section, generate 3 unique QA pairs that a citizen might ask about this law. Each answer must include the section number and a clear, human-friendly explanation.

    Provide questions and answers in 2 English and 1 Bangla language.

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


def generateQA(law_item, language='en')->list[object]:
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


def main():
	with open('data/registration-act.json', 'r') as f:
		law_items = json.load(f)
        
	dataset = []
	for i, law_item in enumerate(law_items):
		print(f"{i+1} - {law_item['section_name_en']}")
		qa_pairs = generateQA(law_item, language='en')
		print(json.dumps(qa_pairs, indent=2))
		for qa in qa_pairs.get('output'):
			dataset.append(qa)

	with open('data/ft_dataset-registration-act.jsonl', 'w') as f:
		for item in dataset:
			f.write(json.dumps(item, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    main()


