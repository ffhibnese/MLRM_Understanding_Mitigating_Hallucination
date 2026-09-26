import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset


class SFTDataset(Dataset):
    def __init__(self, data_path, image_folder, processor):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.image_folder = image_folder
        self.processor = processor
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]

        question = sample['question']
        response = sample['positive_response']
        image_path = os.path.join(self.image_folder, sample['image'])
        
        prompt_text = f"{question} You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE in <answer> </answer> tags."
        
        image = Image.open(image_path).convert('RGB')
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": response}
                ]
            }
        ]
        
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=8192
        )

        input_ids = inputs['input_ids'][0]
        attention_mask = inputs['attention_mask'][0]
        pixel_values = inputs['pixel_values']
        image_grid_thw = inputs.get('image_grid_thw', None)

        labels = input_ids.clone()

        user_text = self.processor.apply_chat_template(
            messages[:1],  
            tokenize=False,
            add_generation_prompt=True 
        )
        user_inputs = self.processor(
            text=[user_text],
            images=[image],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=8192
        )
        user_length = user_inputs['input_ids'].shape[1]

        labels[:user_length] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values,
            'image_grid_thw': image_grid_thw,
            'labels': labels,
        }


def collate_fn(batch, processor=None):

    if processor is not None and hasattr(processor, 'tokenizer'):
        pad_token_id = processor.tokenizer.pad_token_id
    else:
        pad_token_id = 0

    input_ids = torch.nn.utils.rnn.pad_sequence(
        [item['input_ids'] for item in batch],
        batch_first=True,
        padding_value=pad_token_id 
    )

    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [item['attention_mask'] for item in batch],
        batch_first=True,
        padding_value=0
    )

    labels = torch.nn.utils.rnn.pad_sequence(
        [item['labels'] for item in batch],
        batch_first=True,
        padding_value=-100 
    )

    pixel_values = torch.cat([item['pixel_values'] for item in batch], dim=0)
    image_grid_thw_list = [item['image_grid_thw'] for item in batch]
    if image_grid_thw_list[0] is not None:
        image_grid_thw = torch.cat(image_grid_thw_list, dim=0)
    else:
        image_grid_thw = None

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'pixel_values': pixel_values,
        'image_grid_thw': image_grid_thw,
        'labels': labels,
    }

