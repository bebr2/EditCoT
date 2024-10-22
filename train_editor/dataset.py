from torch.utils.data import Dataset
from typing import Dict

from utils import read_hotpotqa_data as read_sft
from utils import llama_instruct_preprocess as data_preprocess



class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path,
        tokenizer,
        model_max_length
    ):
        super(SupervisedDataset, self).__init__()

        self.data = read_sft(data_path)
        
            
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.ignore_index = -100
        
        # 打印第一条数据
        all_len = 0
        max_len = 0
        for i in range(len(self.data)):
            input_ids, _ = data_preprocess(self.data[i], self.tokenizer, self.model_max_length, self.ignore_index)
            len_ = len(input_ids)
            all_len += len_
            max_len = max(max_len, len_)
            
            
        for i in range(3):
            item = self.preprocessing(self.data[i])
            print("【input】:", self.tokenizer.decode(item["input_ids"]))
            labels = []
            for ii, id_ in enumerate(item["labels"]):
                if id_ == self.ignore_index:
                    continue

                labels.append(id_)

                    
            print("【label】:", self.tokenizer.decode(labels))
        print("【max_len】:", max_len)
        print("【avg_len】:", all_len / len(self.data))

    def __len__(self):
        return len(self.data)

    def preprocessing(self, example):
        
        input_ids, labels = data_preprocess(example, self.tokenizer, self.model_max_length, self.ignore_index)

        max_length = self.model_max_length
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
        
        attention_mask = [1] * len(input_ids) + [0] * (max_length - len(input_ids))
        input_ids += [self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0] * (
            max_length - len(input_ids)
        )
        labels += [self.ignore_index] * (
            max_length - len(labels)
        )
        
        return {'input_ids': input_ids, 'labels': labels, 'attention_mask': attention_mask}

    def __getitem__(self, idx) -> Dict[str, list]:
        return self.preprocessing(self.data[idx])