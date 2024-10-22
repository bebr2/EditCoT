import sys

input_path = sys.argv[1]
output_path = sys.argv[2]

import json
data = []

with open(input_path, "r") as f:
    data += json.load(f)
print(len(data))
filter_data = []

for d in data:
    right_steps = d["steps"]
    right_ans = d["answer"]
    # if "unknown" in right_ans.lower() or "none" in right_ans.lower() or "no answer available" in right_ans.lower() or "no answer" in right_ans.lower():
    #     continue
    # if "document" in "".join(right_steps).lower() or "mention" in "".join(right_steps).lower() or "according to" in "".join(right_steps).lower():
    #     continue
    if "insufficient information" in right_ans.lower():
        continue
    if "insufficient information" in "".join(right_steps).lower():
        continue
    sub_data = {"_id": d["_id"], "question": d["question"], "steps": d["steps"], "answer": d["answer"], "changes": []}
    for i, res in enumerate(d["changes"]):
        if not "[STEP]" in res:
            continue
        if not "[ANSWER]" in res:
            continue
        ans = res.split("[ANSWER]")[-1].strip()
        steps = res.split("[ANSWER]")[0].split("[STEP]")
        steps = [s.strip() for s in steps][1:]
        if len(steps) < 3:
            continue
        if abs(len(steps) - len(right_steps)) > 1:
            continue
        if len(ans.split()) - len(right_ans.split()) > 10:
            continue
        if "unknown" in ans.lower() or "none" in ans.lower() or "no answer available" in ans.lower() or "insufficient information" in ans.lower():
            continue
        if "document" in "".join(steps).lower() or "mention" in "".join(steps).lower() or "according to" in "".join(steps).lower() or "insufficient information" in "".join(steps).lower():
            continue
        if steps[i].strip() == right_steps[i].strip():
            continue
        sub_data["changes"].append({"answer": ans, "diff_idx": i, "steps": steps})
    if not sub_data["changes"]:
        continue
    filter_data.append(sub_data)
    
    
print(len(filter_data))
with open(output_path, "w+") as f:
    json.dump(filter_data, f, indent=4)