import os
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria, AutoModel
import torch
import json
from tqdm import tqdm
from editcot import EditCoT
import argparse


def main(args):
    
    model_name = args.model_name
    model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_8bit=False,
                torch_dtype=torch.float16,
                device_map='auto',
                low_cpu_mem_usage=True)
    llmtokenizer = AutoTokenizer.from_pretrained(model_name)

    model_edit = AutoModelForCausalLM.from_pretrained(
                args.editor_path,
                load_in_8bit=False,
                torch_dtype=torch.float16,
                device_map='auto',
                low_cpu_mem_usage=True)

    contriever = AutoModel.from_pretrained(args.retriever_path).cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.retriever_path)

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


    dataset = json.load(open(args.data_path, "r"))
    new_facts = {}
    for d in dataset:
        for r in d["requested_rewrite"]:
            new_facts[f'{r["prompt"].format(r["subject"])} {r["target_true"]["str"]}'] = f'{r["prompt"].format(r["subject"])} {r["target_new"]["str"]}'
    print(len(new_facts))

    all_facts = set()
    for d in dataset:
        for r in d["single_hops"]:
            all_facts.add(r["cloze"] + " " + r["answer"])
    for k in new_facts:
        all_facts.add(k)
    all_facts = list(all_facts)
    embs = get_sent_embeddings(all_facts, contriever, tokenizer)

    tot = 0
    cor = 0

    result = []
    for d in tqdm(dataset):
        tot += 1
        for q in d["questions"]:
            res = EditCoT(model, model_edit, contriever, llmtokenizer, tokenizer, embs, all_facts,q, exist_old_fact=True, old_fact_2_new_fact=new_facts)
            result.append(res)
            ans = res["ans"]
            if ans == d["new_answer"] or ans in d["new_answer_alias"]:
                cor += 1
                break
        print(f'Multi-hop acc = {cor / tot} ({cor} / {tot})')

    print(f'Multi-hop acc = {cor / tot} ({cor} / {tot})')

    with open(args.output_filename, 'w+') as f:
        json.dump(result, f, indent=4)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3-8B-Instruct")
    parser.add_argument("--data_path", type=str, default="./mquake/MQuAKE-CF-3k.json")
    parser.add_argument("--editor_path", type=str, default="../train_editor/output/output_llama")
    parser.add_argument("--retriever_path", type=str, default="facebook/contriever-msmarco")
    parser.add_argument("--output_filename", type=str, default="./mquake/output/output.json")
    parser.add_argument("--max_iter", type=int, default=4)
    args = parser.parse_args()
    
    main(args)