# %%


import sys
import json
import os
model_path = sys.argv[1]
source_path = sys.argv[2]
rag_output_path = sys.argv[3]
prompt_path = sys.argv[4]


import torch
from tqdm import tqdm

with open(f"{source_path}/hotpot_test_fullwiki_v1.json") as f:
    data = json.load(f)

# %%
with open(f"{source_path}/hotpot_dev_fullwiki_v1.json") as f:
    data += json.load(f)

# %%
with open(f"{prompt_path}/rag_answer.txt") as f:
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
def get_answer(docs, q):
    generation_config = dict(
                            do_sample=False,
                            num_beams=1,
                            max_new_tokens=512,
                            return_dict_in_generate=True,
                            output_hidden_states=True,
                            eos_token_id=terminators,
                            output_scores = True)
    doc_str = ""
    for doc in docs:
        doc_str += doc[0] + "\n" + "".join(doc[1]) + "\n\n"
    input_ids = llmtokenizer.apply_chat_template(
        [{"role": "user", "content": f"{doc_str}{prompt} {q}"}],
        add_generation_prompt=True,
    )
    input_ids += llmtokenizer.encode("[STEP]", add_special_tokens=False)
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
    output_ = "[STEP] " + llmtokenizer.decode(response, skip_special_tokens=True).strip()
    # print(f"【Output】: {output_}")
    # print("---------------------------")
    return output_

# %%
qa = []
# for d in tqdm(data[:10]):
for d in tqdm(data):
    question = d["question"]
    response = get_answer(d["context"][:5], question)
    qa.append({"_id": d["_id"], "question": question, "response": response})

with open(rag_output_path, "w+") as f:
    json.dump(qa, f, indent=4)


