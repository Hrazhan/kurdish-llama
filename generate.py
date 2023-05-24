import sys
import torch
from peft import PeftModel, PeftModelForCausalLM, LoraConfig
import transformers
import gradio as gr
import argparse
import warnings
import os
from utils import SteamGenerationMixin
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="decapoda-research/llama-7b-hf")
parser.add_argument("--lora_path", type=str, default="razhan/kurdish-llama-lora")
parser.add_argument("--use_typewriter", type=int, default=1)
parser.add_argument("--use_local", type=int, default=1)
args = parser.parse_args()
print(args)
tokenizer = LlamaTokenizer.from_pretrained(args.model_path)

LOAD_8BIT = True
BASE_MODEL = args.model_path
LORA_WEIGHTS = args.lora_path



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device == "cuda":
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_8bit=LOAD_8BIT,
        torch_dtype=torch.float16,
        device_map={"": 0},
    )
    model = SteamGenerationMixin.from_pretrained(
        model, LORA_WEIGHTS, torch_dtype=torch.float16, device_map={"": 0}
    )
else:
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL, device_map={"": device}, low_cpu_mem_usage=True
    )
    model = SteamGenerationMixin.from_pretrained(
        model,
        LORA_WEIGHTS,
        device_map={"": device},
    )


def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
                ### Instruction:
                {instruction}
                ### Input:
                {input}
                ### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
                ### Instruction:
                {instruction}
                ### Response:"""


if not LOAD_8BIT:
    model.half()  # seems to fix bugs for some users.

model.eval()
if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)


def evaluate(
    input,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    max_new_tokens=128,
    min_new_tokens=1,
    repetition_penalty=2.0,
    **kwargs,
):
    prompt = generate_prompt(input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        max_new_tokens=max_new_tokens, # max_length=max_new_tokens+input_sequence
        min_new_tokens=min_new_tokens, # min_length=min_new_tokens+input_sequence
        **kwargs,
    )
    with torch.no_grad():
        if args.use_typewriter:
            for generation_output in model.stream_generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=False,
                repetition_penalty=float(repetition_penalty),
            ):
                outputs = tokenizer.batch_decode(generation_output)
                show_text = "\n--------------------------------------------\n".join(
                    [output.split("### Response:")[1].strip().replace('�','')+" ▌" for output in outputs]
                )
                # if show_text== '':
                #     yield last_show_text
                # else:
                yield show_text
            yield outputs[0].split("### Response:")[1].strip().replace('�','')
        else:
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=False,
                repetition_penalty=1.3,
            )
            output = generation_output.sequences[0]
            output = tokenizer.decode(output).split("### Response:")[1].strip()
            print(output)
            yield output


gr.Interface(
    fn=evaluate,
    inputs=[
        gr.components.Textbox(
            lines=2, label="Input", placeholder="دەربارەی Python شتێکم پێبڵێ"
        ),
        gr.components.Slider(minimum=0, maximum=1, value=0.1, label="Temperature"),
        gr.components.Slider(minimum=0, maximum=1, value=0.75, label="Top p"),
        gr.components.Slider(minimum=0, maximum=100, step=1, value=40, label="Top k"),
        gr.components.Slider(minimum=1, maximum=10, step=1, value=4, label="Beams Number"),
        gr.components.Slider(
            minimum=1, maximum=2000, step=1, value=256, label="Max New Tokens"
        ),
        gr.components.Slider(
            minimum=1, maximum=300, step=1, value=1, label="Min New Tokens"
        ),
        gr.components.Slider(
            minimum=0.1, maximum=10.0, step=0.1, value=2.0, label="Repetition Penalty"
        ),
    ],
    outputs=[
        gr.inputs.Textbox(
            lines=25,
            label="Output",
        )
    ],
    title="Kurdish LLaMa",
    description="Kurdish LLaMa is fine-tuned LLaMa model on the set of instruction provided by the Alpaca project with output of GPT4",
    article = "<p style='text-align: center'> Made with ❤️ by <a href='https://twitter.com/RazhanHameed'>Razhan Hameed</a></p>"
).queue().launch(share=True)