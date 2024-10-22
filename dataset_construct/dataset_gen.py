# %%
import sys
import json
import os
import numpy as np

model_path = sys.argv[1]
source_path = sys.argv[2]
dataset_output_path = sys.argv[3]
prompt_path = sys.argv[4]
ratio = float(sys.argv[5])

import torch
from tqdm import tqdm

# %%
with open(f"{prompt_path}/verify.txt") as f:
    verify_prompt = f.read().strip()
    
with open(f"{prompt_path}/conflict.txt") as f:
    conflict_prompt = f.read().strip()
    
with open(f"{prompt_path}/rewrite.txt") as f:
    rewrite_prompt = f.read().strip()


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
def is_consistent(q, a1, a2):
    generation_config = dict(
                            do_sample=False,
                            num_beams=1,
                            max_new_tokens=1,
                            return_dict_in_generate=True,
                            output_hidden_states=True,
                            eos_token_id=terminators,
                            output_scores = True)
    ansid = [llmtokenizer.encode("A", add_special_tokens=False)[0],
    llmtokenizer.encode("B", add_special_tokens=False)[0]]
    input_ids = llmtokenizer.apply_chat_template(
        [{"role": "user", "content": verify_prompt.format(q, a1, a2)}],
        add_generation_prompt=True,
    )
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(model.device)
    # print(f"【Input】: {llmtokenizer.decode(input_ids[0], skip_special_tokens=False)}")
    # print("---------------------------")
    
    outputs = model.generate(
        input_ids,
        pad_token_id=llmtokenizer.eos_token_id,
        **generation_config
    )
    logits = outputs.scores[0][0]
    logits = logits.float().cpu().detach()
    choices_logits = logits[ansid].numpy()
    # 算softmax

    assert not (np.any(np.isinf(choices_logits)) or np.any(np.isnan(choices_logits)))
    choices_p = np.exp(choices_logits) / np.sum(np.exp(choices_logits))
    # print(f"【Consistent】: {choices_p[1]}")
    if choices_p[1] > ratio:
        return False
    else:
        return True

def is_conflict(s1, s2):
    generation_config = dict(
                            do_sample=False,
                            num_beams=1,
                            max_new_tokens=1,
                            return_dict_in_generate=True,
                            output_hidden_states=True,
                            eos_token_id=terminators,
                            output_scores = True)
    ansid = [llmtokenizer.encode("A", add_special_tokens=False)[0],
    llmtokenizer.encode("B", add_special_tokens=False)[0],
    llmtokenizer.encode("C", add_special_tokens=False)[0]]
    input_ids = llmtokenizer.apply_chat_template(
        [{"role": "user", "content": conflict_prompt.format(s1, s2)}],
        add_generation_prompt=True,
    )
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(model.device)
    # print(f"【Input】: {llmtokenizer.decode(input_ids[0], skip_special_tokens=False)}")
    # print("---------------------------")
        
    outputs = model.generate(
        input_ids,
        pad_token_id=llmtokenizer.eos_token_id,
        **generation_config
    )
    logits = outputs.scores[0][0]
    logits = logits.float().cpu().detach()
    choices_logits = logits[ansid].numpy()
    # 算softmax

    assert not (np.any(np.isinf(choices_logits)) or np.any(np.isnan(choices_logits)))
    choices_p = np.exp(choices_logits) / np.sum(np.exp(choices_logits))
    # print(f"【Conflict】: {choices_p[0]}")
    if choices_p[0] > ratio:
        return True
    else:
        return False

def rewrite(s):
    generation_config = dict(
                            do_sample=False,
                            num_beams=1,
                            max_new_tokens=128,
                            return_dict_in_generate=True,
                            output_hidden_states=True,
                            eos_token_id=terminators,
                            output_scores = True)

    input_ids = llmtokenizer.apply_chat_template(
        [{"role": "user", "content": rewrite_prompt.format(s)}],
        add_generation_prompt=True,
    )
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
    output_ = llmtokenizer.decode(response, skip_special_tokens=True).strip()
    # print(f"【Output】: {output_}")
    # print("---------------------------")
    return output_


with open(source_path, "r") as f:
    data = json.load(f)

qa = []
for d in tqdm(data):
    right_ans = d["answer"]
    q = d["question"]
    right_steps = d["steps"]
    for change in reversed(d["changes"]):
        steps = change["steps"]
        if change["diff_idx"] >= len(right_steps):
            continue
        if change["diff_idx"] >= len(steps):
            continue
        ans = change["answer"]
        res = is_consistent(q, right_ans, ans)
        if not res:
            
            s1 = right_steps[change["diff_idx"]]
            s2 = change["steps"][change["diff_idx"]]
            
            # res = 
            
            if is_conflict(s1, s2):
                # rewrite
                s1_rewrite = rewrite(s1)
                s2_rewrite = rewrite(s2)
                if is_conflict(s1_rewrite, s2_rewrite):
                    qa.append({"_id": d["_id"], "question": q, "right_answer": right_ans, "right_steps": right_steps, "changed_steps": steps, "changed_answer": ans, "diff_idx": change["diff_idx"], "right_fact": s1_rewrite, "changed_fact": s2_rewrite})
                
            
    print(len(qa))
            
with open(dataset_output_path, "w+") as f:
    json.dump(qa, f, indent=4)
                


