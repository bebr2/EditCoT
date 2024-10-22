# %%
import sys
import json
import os

model_path = sys.argv[1]
source_path = sys.argv[2]
pg_output_path = sys.argv[3]
prompt_path = sys.argv[4]

import torch
from tqdm import tqdm

# %%
with open(f"{prompt_path}/answer.txt") as f:
    prompt = f.read().strip()
print(prompt)

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria
model = AutoModelForCausalLM.from_pretrained(
            model_path,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map='auto',
            low_cpu_mem_usage=True)
llmtokenizer = AutoTokenizer.from_pretrained(model_path)

terminators = [
    llmtokenizer.eos_token_id,
    llmtokenizer.convert_tokens_to_ids("<|eot_id|>")
]
terminators = [t for t in terminators if t is not None]
# %%
def get_answer(steps, q, token):
    generation_config = dict(
                            do_sample=False,
                            num_beams=1,
                            max_new_tokens=512,
                            return_dict_in_generate=True,
                            output_hidden_states=True,
                            eos_token_id=terminators,
                            output_scores = True)
    cot_str = "[STEP] "
    for step in steps:
        cot_str += step + "\n[STEP] "
    cot_str += token
    input_ids = llmtokenizer.apply_chat_template(
        [{"role": "user", "content": f"{prompt} {q}"}],
        add_generation_prompt=True,
    )
    input_ids += llmtokenizer.encode(cot_str, add_special_tokens=False)
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(model.device)
    # print(f"【Input】: {llmtokenizer.decode(input_ids[0], skip_special_tokens=False)}")
    # print("---------------------------")
        
    outputs = model.generate(
        input_ids,
        pad_token_id=llmtokenizer.eos_token_id,
        **generation_config
    )
    # print(response)
    response = outputs.sequences[0][input_ids.shape[-1]:]
    # print("【Response】: ", response)
    output_ = f"{cot_str.strip()} " + llmtokenizer.decode(response, skip_special_tokens=True).strip()
    # print(f"【Output】: {output_}")
    # print("---------------------------")
    return output_

with open(source_path, "r") as f:
    data = json.load(f)

qa = []
for d in tqdm(data):
# for d in tqdm(data[:3]):
    sub_data = {"_id": d["_id"], "question": d["question"], "steps": d["steps"], "answer": d["answer"], "changes": []}
    for i in range(len(d["steps"])):
        response = get_answer(d["steps"][:i], d["question"], d["steps"][i].split()[0])
        sub_data["changes"].append(response)
    qa.append(sub_data)

with open(pg_output_path, "w+") as f:
    json.dump(qa, f, indent=4)
