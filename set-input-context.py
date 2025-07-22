import json
import re

from utils.util import convertBn2En

input_files = ['input-data/state-aquisition.json', 'input-data/registration-act.json',
               'input-data/the-transfer-of-property-act.json']
output_files = ['output-data/state-aquisition.json', 'output-data/registration-act.json',
                'output-data/the-transfer-of-property-act.json']

def getSections(law_reference:str)->list[str]:
	if "ধারা" in law_reference or "Sections" in law_reference:
		section = law_reference.split("Sections")[-1] if "Sections" in law_reference else law_reference.split("ধারা")[-1]
	else:
		section = law_reference.split(", ")[-1]

	section = section.split("Section")[-1]
	section = re.split(r"ও|and|,|&", section)
	section = [re.sub(r'\([^)]*\)', "", convertBn2En(s.strip())) for s in section]
	section = [s.split('-') for s in section]
	sections = list(set([s for se in section for s in se if s]))
	return sections


def main():
	process_idx = 2
	with open(output_files[process_idx], 'r') as f:
		qa_pairs = json.load(f)
		
	with open(input_files[process_idx], 'r') as f:
		main_data:list[dict] = json.load(f)

	mod_pairs = []

	for qa_pair in qa_pairs:
		law_reference: str = qa_pair['law_reference']
		sections = getSections(law_reference)
		print(law_reference, "\t--->", sections)
		content = []
		for sec in sections:
			act = list(filter(lambda a: a.get('section_no_en')==sec, main_data))
			content.append(act[0]['content'])
		
		qa_pair['input_context'] = "\n---\n".join(content)

		mod_pairs.append(qa_pair)
	
	print(len(mod_pairs)," Pairs are ready to write...", end=" ")
	with open(output_files[process_idx], 'w') as f:
		f.write(json.dumps(mod_pairs, ensure_ascii=False))
		print("Done")

if __name__ == "__main__":
	main()
