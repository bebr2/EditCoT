import os
import json
import random
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

import sys

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria
from transformers.generation.stopping_criteria import StoppingCriteriaList

prompt_path = "/home/wangchangyue/EditCoT-Code/all_prompts/edticot"

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords_ids:list):
        self.keywords = keywords_ids
        # print(self.keywords)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        
        for l in self.keywords:
            if input_ids[0][-len(l):].tolist() == l:
                return True
        return False



def editcot(cot, q, new_knowledge):
    s = f'Question: {q}\nOld Thoughts: {cot}\nNew knowledge: {new_knowledge}\nPlease give me the new chain of thought based on the new knowledge.'
    msg = [
        {"role": "system", "content": "You can edit the following chains of thought based on the new knowledge."},
        {"role": "user", "content": s},
    ]
    return msg

with open(f"{prompt_path}/finalanswer.json") as f:
    finalanswer_prompt = json.load(f)
    
def finalanswer(cot, q):
    history = finalanswer_prompt + [{"role": "user", "content": f"Question: {q}\n[New Thoughts]: {cot}"}]
    return history

# def answer_question(q):
#     history = qa_prompt + [{"role": "user", "content": f"Question: {q}"}]
#     return history

def get_final_answer(q, cot, stopping_criteria, llmtokenizer, model):
    generation_config = dict(
                            do_sample=False,
                            num_beams=1,
                            max_new_tokens=50,
                            return_dict_in_generate=True,
                            output_hidden_states=True,
                            output_scores = True)
    input_ids = llmtokenizer.apply_chat_template(
        finalanswer(cot, q),
        add_generation_prompt=True,
    )
    # print("GET FINAL ANSWER")
    # print(f"【Input】: {llmtokenizer.decode(input_ids[0]).split('<|start_header_id|>user<|end_header_id|>')[-1]}")
    
    input_ids += llmtokenizer.encode("Answer from [New Thoughts]:", add_special_tokens=False)
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(model.device)
    
    outputs = model.generate(
        input_ids,
        stopping_criteria=stopping_criteria,
        pad_token_id=llmtokenizer.eos_token_id,
        attention_mask=input_ids.ne(llmtokenizer.eos_token_id),
        **generation_config
    )
    # print(response)
    response = outputs.sequences[0][input_ids.shape[-1]:]   
    # print("【Response】: ", response)
    output_ = llmtokenizer.decode(response, skip_special_tokens=True)
    if output_.endswith("<|im_end|>"):
        output_ = output_[:-len("<|im_end|>")].strip()
    if output_.endswith("<|eot_id|>"):
        output_ = output_[:-len("<|eot_id|>")].strip()
    # print(f"【Output】: {output_}")
    # print("---------------------------")
    return output_


# %%

with open(f"{prompt_path}/recot.json", "r") as f:
    recot_prompt = json.load(f)

def recot(q, a):
    history = recot_prompt + [{"role": "user", "content": f"Question: {q}\nAnswer: {a}\nPlease give the chain of thought based on the question and answer pairs above."}]
    return history

with open(f"{prompt_path}/qa.json", "r") as f:
    qa_prompt = json.load(f)

def answer_question(q):
    history = qa_prompt + [{"role": "user", "content": f"Question: {q}"}]
    return history


def get_answer(q, llmtokenizer, model, stopping_criteria):
    generation_config = dict(
                            do_sample=False,
                            num_beams=1,
                            max_new_tokens=256,
                            return_dict_in_generate=True,
                            output_hidden_states=True,
                            output_scores = True)
    input_ids = llmtokenizer.apply_chat_template(
        answer_question(q),
        add_generation_prompt=True,
    )
    input_ids += llmtokenizer.encode("Answer:", add_special_tokens=False)
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(model.device)
    outputs = model.generate(
        input_ids,
        stopping_criteria=stopping_criteria,
        pad_token_id=llmtokenizer.eos_token_id,
        attention_mask=input_ids.ne(llmtokenizer.eos_token_id),
        **generation_config
    )
    response = outputs.sequences[0][input_ids.shape[-1]:]
    output_ = llmtokenizer.decode(response, skip_special_tokens=True)
    if output_.endswith("\n\nQuestion"):
        output_ = output_[:-len("\n\nQuestion")].strip()
    if output_.endswith("<|im_end|>"):
        output_ = output_[:-len("<|im_end|>")].strip()
    if output_.endswith("<|eot_id|>"):
        output_ = output_[:-len("<|eot_id|>")].strip()
    return output_
    
def get_recot(q, a, llmtokenizer, model, stopping_criteria):
    generation_config = dict(
                            do_sample=False,
                            num_beams=1,
                            max_new_tokens=256,
                            return_dict_in_generate=True,
                            output_hidden_states=True,
                            output_scores = True)
    input_ids = llmtokenizer.apply_chat_template(
        recot(q, a),
        add_generation_prompt=True,
    )
    input_ids += llmtokenizer.encode("Thoughts:", add_special_tokens=False)
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(model.device)
    outputs = model.generate(
        input_ids,
        stopping_criteria=stopping_criteria,
        pad_token_id=llmtokenizer.eos_token_id,
        attention_mask=input_ids.ne(llmtokenizer.eos_token_id),
        **generation_config
    )
    response = outputs.sequences[0][input_ids.shape[-1]:]
    output_ = llmtokenizer.decode(response, skip_special_tokens=True)
    if output_.endswith("\n\nQuestion"):
        output_ = output_[:-len("\n\nQuestion")].strip()
    if output_.endswith("<|im_end|>"):
        output_ = output_[:-len("<|im_end|>")].strip()
    if output_.endswith("<|eot_id|>"):
        output_ = output_[:-len("<|eot_id|>")].strip()
    return output_


with open(f"{prompt_path}/detection.json", "r") as f:
    edit_cot_prompt = json.load(f)
        
import numpy as np

def get_verify(cot, q, new_knowledge, llmtokenizer, model, stopping_criteria):
    generation_config = dict(
                            do_sample=False,
                            num_beams=1,
                            max_new_tokens=1,
                            return_dict_in_generate=True,
                            output_hidden_states=True,
                            output_scores = True)
    ansid = [llmtokenizer.encode("Contradict", add_special_tokens=False)[0],
    llmtokenizer.encode("Support", add_special_tokens=False)[0],
    llmtokenizer.encode("Unrelated", add_special_tokens=False)[0]]
    
    ansid2 = [llmtokenizer.encode("Modify", add_special_tokens=False)[0],
    llmtokenizer.encode("Related", add_special_tokens=False)[0],
    llmtokenizer.encode("Irrelevant", add_special_tokens=False)[0]]
    s = f'Question: {q}\nOld Thoughts: {cot}\nNew knowledge: {new_knowledge}\nPlease give me the new chain of thought based on the new knowledge.'
    msg = edit_cot_prompt + [{"role": "user", "content": s}]
    input_ids = llmtokenizer.apply_chat_template(
        msg,
        add_generation_prompt=True,
    )
    input_ids += llmtokenizer.encode("Relevance of knowledge and chain of thought: ", add_special_tokens=False)
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(model.device)
    outputs = model.generate(
        input_ids,
        stopping_criteria=stopping_criteria,
        pad_token_id=llmtokenizer.eos_token_id,
        attention_mask=input_ids.ne(llmtokenizer.eos_token_id),
        **generation_config
    )
    logits = outputs.scores[0][0]

    logits = logits.float().cpu().detach()
    choices1_logits = logits[ansid]
    choices2_logits = logits[ansid2]
    choices_logits = (choices1_logits + choices2_logits).numpy()
    assert not (np.any(np.isinf(choices_logits)) or np.any(np.isnan(choices_logits)))
    ans = {0: "Contradict", 1: "Support", 2: "Unrelated"}[np.argmax(choices_logits)]
    return ans

def get_editcot(cot, q, new_knowledge, llmtokenizer, model_edit, stopping_criteria):
    generation_config = dict(
                            do_sample=False,
                            num_beams=1,
                            max_new_tokens=256,
                            return_dict_in_generate=True,
                            output_hidden_states=True,
                            output_scores = True)

    input_ids = llmtokenizer.apply_chat_template(
        editcot(cot, q, new_knowledge),
        add_generation_prompt=True,
    )
    input_ids += llmtokenizer.encode("New Thoughts: ", add_special_tokens=False)
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(model_edit.device)
    outputs = model_edit.generate(
        input_ids,
        stopping_criteria=stopping_criteria,
        pad_token_id=llmtokenizer.eos_token_id,
        attention_mask=input_ids.ne(llmtokenizer.eos_token_id),
        **generation_config
    )
    response = outputs.sequences[0][input_ids.shape[-1]:]
    output_ = llmtokenizer.decode(response, skip_special_tokens=True)
    if output_.endswith("\n\nQuestion"):
        output_ = output_[:-len("\n\nQuestion")].strip()
    if output_.endswith("<|im_end|>"):
        output_ = output_[:-len("<|im_end|>")].strip()
    if output_.endswith("<|eot_id|>"):
        output_ = output_[:-len("<|eot_id|>")].strip()
    return output_



# %%
def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

def get_sent_embeddings(sents, contriever, tok, BSZ=32):    
    all_embs = []
    for i in tqdm(range(0, len(sents), BSZ)):
        sent_batch = sents[i:i+BSZ]
        inputs = tok(sent_batch, padding=True, truncation=True, return_tensors='pt').to("cuda")
        with torch.no_grad():
            outputs = contriever(**inputs)
            embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
        all_embs.append(embeddings.cpu())
    all_embs = torch.vstack(all_embs)
    return all_embs

def retrieve_facts(query, fact_embs, contriever, tok, k=1):
    inputs = tok([query], padding=True, truncation=True, return_tensors='pt').to("cuda")
    with torch.no_grad():
        outputs = contriever(**inputs)
        query_emb = mean_pooling(outputs[0], inputs['attention_mask']).cpu()
    sim = (query_emb @ fact_embs.T)[0]
    knn = sim.topk(k, largest=True)
    return knn.indices, knn.values

def EditCoT(model, model_edit, contriever, llmtokenizer, tokenizer, fact_embs, fact_docs, q, max_iter=4, exist_old_fact=True, old_fact_2_new_fact=None):
    
    stop_list = [
        [100],
        llmtokenizer.encode("\n\nQuestion", add_special_tokens=False),
        llmtokenizer.encode(".\n\nQuestion", add_special_tokens=False),
        llmtokenizer.encode("<|im_end|>", add_special_tokens=False),
        llmtokenizer.encode("<|eot_id|>", add_special_tokens=False),
    ]
    stop_criteria = KeywordsStoppingCriteria(stop_list)
    stopping_criteria = StoppingCriteriaList([stop_criteria])
    
    if exist_old_fact:
        assert old_fact_2_new_fact is not None
    fact_ids_list = []
    gen = get_answer(q, llmtokenizer, model, stopping_criteria)
    old_answer = gen.split("Answer:")[-1].strip()

    gen = get_recot(q, old_answer, llmtokenizer, model, stopping_criteria)
    old_cot = gen.split("Thoughts:")[-1].strip()
    original_cot = old_cot
    record = f"【Question】: {q}\n【Old Thoughts】: {old_cot}\n【Old Answer】: {old_answer}\n"
    for i in range(max_iter):
        find_new_fact = False
        sentences = old_cot.split("\n")
        new_fact = None
        if i == 0:
            sentences = [q] + sentences
        for s in sentences:
            if s.strip() == "":
                continue
            fact_ids, fact_value = retrieve_facts(s, fact_embs, contriever, tokenizer)
            
            if exist_old_fact:
                new_fact = old_fact_2_new_fact.get(fact_docs[fact_ids[0]])
            else:
                new_fact = fact_docs[fact_ids[0]]
            
            if new_fact is not None and fact_ids[0] not in fact_ids_list:
                fact_ids_list.append(fact_ids[0])
                find_new_fact = True
                record += f"【Sentence】: {s}\n【New Fact】: {new_fact}\n"
                break
        if not find_new_fact:
            record += f"【Iteration】: {i}\n【Find New Fact】: False\n"
            break
            
        gen = get_verify(old_cot, q, new_fact, llmtokenizer, model, stopping_criteria)
        # old_cot = gen.split("New Thoughts: ")[-1].strip()
        record += f"【Iteration】: {i}\n【Verify】: {gen}\n"
        if "Support" in gen or "Unrelated" in gen:
            break
        else:
            gen = get_editcot(old_cot, q, new_fact, llmtokenizer, model_edit, stopping_criteria)
            old_cot = gen.split("New Thoughts:")[-1].strip()
            record += f"【EditCoT】: {old_cot}\n"
        
    gen = get_final_answer(q, old_cot, stopping_criteria, llmtokenizer, model)
    ans = gen.split("Answer from [New Thoughts]:")[-1].strip().split("\n")[0]
    record += f"【Final Answer】: {ans}\n"
    
    return {"q": q, "oa": old_answer, "ocot": original_cot, "ncot": old_cot, "ans": ans, "record": record}