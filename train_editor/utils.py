import json

def read_hotpotqa_data(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    new_data = []
    for d in data:
        thoughts = "\n".join(d["changed_steps"])
        new_thoughts = "\n".join(d["right_steps"])
        new_knowledge = d["right_fact"]
        s = f'Question: {d["question"]}\nOld Thoughts: {thoughts}\nNew knowledge: {new_knowledge}\nPlease give me the new chain of thought based on the new knowledge.'
        new_data.append([
            {"role": "system", "content": "You can edit the following chains of thought based on the new knowledge."},
            {"role": "user", "content": s},
            {"role": "assistant", "content": f"New Thoughts: {new_thoughts}"}
        ])
    return new_data

def llama_instruct_preprocess(msg, tokenizer, max_len, ignore_index=-100):
    ids = tokenizer.apply_chat_template(
        msg,
        tokenize=True,
        add_generation_prompt=False,
    )
    start = 0
    for i in range(len(ids) - 1, -1, -1):
        if ids[i-3:i+1] == [128006, 78191, 128007, 271]:
            start = i + 1
            break
    labels = [ignore_index] * start + ids[start:]
    
    return ids, labels