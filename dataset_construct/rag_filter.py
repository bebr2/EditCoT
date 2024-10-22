import json
import sys

input_path = sys.argv[1]
output_path = sys.argv[2]

with open(input_path, "r") as f:
    data = json.load(f)
    
filter_data = []

for d in data:
    res = d["response"]
    # 能提取出[STEP]和[ANSWER]
    # steps长度大于等于2
    # [ANSWER]的长度不超过50个words
    if not "[STEP]" in res:
        continue
    if not "[ANSWER]" in res:
        continue
    ans = res.split("[ANSWER]")[-1].strip()
    steps = res.split("[ANSWER]")[0].split("[STEP]")
    steps = [s.strip() for s in steps]
    if len(steps) < 3:
        continue
    if len(ans.split()) > 20:
        continue
    
    if "unknown" in ans.lower() or "none" in ans.lower() or "no answer available" in ans.lower() or "no answer" in ans.lower():
        continue
    if "document" in "".join(steps).lower() or "mention" in "".join(steps).lower() or "according to" in "".join(steps).lower():
        continue
    
    filter_data.append({"_id": d["_id"], "question": d["question"], "answer": ans, "steps": steps[1:]})
    
    
print(len(filter_data))
with open(output_path, "w+") as f:
    json.dump(filter_data, f, indent=4)