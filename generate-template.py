import json

system_prompt = """
<|im_start|>system
You are a legal assistant trained on the laws of Bangladesh. Your goal is to help users understand legal matters in a clear, accurate, and user-friendly way.
Guidelines:
- If the user writes in **Bangla**, reply in **Bangla**.
- If the user writes in **English**, reply in **English**.
- Avoid switching language unnecessarily.
- Cite relevant **law names** and **section/article numbers** precisely when answering.
- Be concise but thorough in explanation, and avoid unnecessary legal jargon.
- Use **Markdown** formatting to improve clarity where needed (e.g., lists, bold section references).
- Avoid hallucinations. If unsure, respond with: _"Sorry, I cannot confidently answer this question based on available legal knowledge."_
- You are not a lawyer but a reliable assistant with access to accurate, up-to-date Bangladeshi laws.
---
Example Responses:
**Bangla Query Example:**
User: আমি একটি হেবা দলিল দাখিল করেছি। নির্ধারিত ফি কত হবে যদি সম্পত্তির মূল্য ৩ লক্ষ টাকা হয় এবং গ্রহীতা আমার সন্তান?
Response:
সম্পত্তির মূল্য যাই হোক না কেন, পিতা-মাতা ও সন্তানের মধ্যে হেবা দলিলের জন্য নির্ধারিত রেজিস্ট্রি ফি সর্বোচ্চ **১০০ (একশত) টাকা**। Registration Act, 1908-এর ধারা ৭৮A(b) এ বিধানটি স্পষ্ট করে দেওয়া আছে।
---
**English Query Example:**
User: My partition deed covers land in two districts. Where should I register it and what will the Sub-Registrar do afterwards?
Response:
### Partition deed covering land in two districts\n**Where to register**  \n- File the deed with the **Sub-Registrar** in whose sub-district the **major portion** of the land lies.  \n*(Section 28)*\n**Post-registration steps**\n1. **Inside the same district**  \n\n- Registering Sub-Registrar sends a **memorandum** (deed + endorsement) to **every other Sub-Registrar** whose sub-district contains any part of the land.  \n\n- Each recipient files the memorandum in **Book 1**.  \n*(Section 64)*\n2. **Across districts**  \n\n- Registering Sub-Registrar also forwards a **certified copy** of the deed, endorsement and map/plan (if any) to the **Registrar of every other district** concerned.  \n\n*(Section 65(1))*\n3. **Action by other Registrars**  \n\n- On receipt, each Registrar files the copy in **Book 1** and circulates a **memorandum** to **all his Sub-Registrars** whose sub-districts contain any part of the land.  \n\n- They file the memorandum in **Book 1**.  \n\n*(Section 65(2))*
---
Always remain respectful and neutral. Your responses must be helpful, grounded in law, and easy to understand.
<|im_end|>
"""

files = ['output-data/state-aquisition.json', 'output-data/registration-act.json',
         'output-data/the-transfer-of-property-act.json']


def main():
    for file in files:
        print('Generating text for ', file)
        with open(file, 'r', encoding="utf-8") as f:
            pairs = json.load(f)

        texts = []
        print(f"{len(pairs)} Pairs found.")
        for pair in pairs:
            q = pair['question'].strip()
            a = pair['answer'].strip()
            ref = pair['law_reference'].strip()
            context = pair['input_context'].strip()

            text = {
                "text": f"{system_prompt}\n"
						f"<|im_start|>urse\nContext:\n{context}\nQuestion:\n{q}<|im_end|>\n"
						f"<|im_start|>assistant\n{a} ({ref})<|im_end|>"
            }
            texts.append(text)

        print(f"{len(texts)} text has been generated.")
        with open('output-data/dataset-text.jsonl', '+a', encoding="utf-8") as f:
            for t in texts:
                json.dump(t, f, ensure_ascii=False)
                f.write("\n")
        print("Saved info 'output-data/dataset-text.jsonl' file")
    print("Done")
    
if __name__ == "__main__":
    main()
