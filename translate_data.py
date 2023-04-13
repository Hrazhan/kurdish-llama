from transformers import pipeline
import json
import re
from tqdm.auto import tqdm
import sys

TRANSLATION_MODEL = "facebook/nllb-200-distilled-600M"
translator = pipeline('translation', model=TRANSLATION_MODEL, tokenizer=TRANSLATION_MODEL, src_lang="eng_Latn", tgt_lang="ckb_Arab", device=0)

def has_code_block(text):
    # Check if there is a code block delimited by three backticks in the text
    return re.search(r'```.*?```', text, re.DOTALL) is not None


def replace_code_with_placeholder(text):
    # Find all occurrences of code blocks delimited by three backticks
    code_blocks = re.findall(r'```.*?```', text, re.DOTALL)
    
    # Replace each code block with a placeholder
    for i, code_block in enumerate(code_blocks):
        text = text.replace(code_block, f'{{cod_blok_{i}}}')
        
    return text, code_blocks

def replace_placeholder_with_code(text, code_blocks):
    # Replace each placeholder with its corresponding code block
    for i, code_block in enumerate(code_blocks):
        text = text.replace(f'{{cod_blok_{i}}}', code_block)
        
    return text


def translate(text, length=1200):
    output = translator(text, max_length=length)[0]['translation_text']
    return output

def helper(text):
    if has_code_block(text):
        text, code_blocks = replace_code_with_placeholder(text)
        text = replace_placeholder_with_code(translate(text), code_blocks)
    else:
        text = translate(text)
    return text
    


def dequote(s):
    """
    If a string has single or double quotes around it, remove them.
    Make sure the pair of quotes match.
    If a matching pair of quotes is not found,
    or there are less than 2 characters, return the string unchanged.
    """
    if (len(s) >= 2 and s[0] == s[-1]) and s.startswith(("'", '"')):
        return s[1:-1]
    return s





def main(input_file, output_file):
    # Open the input JSON file and load the data
    with open(input_file, 'r') as f:
        data = json.load(f)

    print(f"Total number of instructions: {len(data)}")
    
    for item in tqdm(data, desc="Processing instructions"):
        item['instruction'] = helper(dequote(item['instruction']))
        item['input'] = helper(dequote(item['input']))
        item['output'] = helper(dequote(item['output'])) 
    

    # # Write the translated data to a new JSON file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 translate_data.py <input_file> <output_file>")
        exit(1)
    main(sys.argv[1], sys.argv[2])