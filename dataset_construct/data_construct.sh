gpu_id="0"

project_path="/xxx/xxx/EditCoT"
model_path="/path/to/model"


prompt_path=$project_path"/all_prompts/trainset"
root_path=$project_path"/dataset_construct"
source_path=$root_path"/raw"
output_path=$root_path"/output"
conflict_ratio=0.6

# RAG QA

CUDA_VISIBLE_DEVICES=$gpu_id python $root_path/rag_qa.py $model_path $source_path $output_path"/rag_qa.json" $prompt_path

# Filter 1

CUDA_VISIBLE_DEVICES=$gpu_id python $root_path/rag_filter.py $output_path"/rag_qa.json" $output_path"/rag_qa_filtered.json"

# Prefix guided generation

CUDA_VISIBLE_DEVICES=$gpu_id python $root_path/prefix_guided.py $model_path $output_path"/rag_qa_filtered.json" $output_path"/pg_qa.json" $prompt_path

# Filter 2

CUDA_VISIBLE_DEVICES=$gpu_id python $root_path/pg_filter.py $output_path"/pg_qa.json" $output_path"/pg_qa_filtered.json"

# Construct dataset

CUDA_VISIBLE_DEVICES=$gpu_id python $root_path/dataset_gen.py $model_path $output_path"/pg_qa_filtered.json" $output_path"/dataset.json" $prompt_path $conflict_ratio
    


