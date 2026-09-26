import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset


QWEN_PROMPT_TEMPLATE = """{question} You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE in <answer> </answer> tags."""


def load_image(image_path):
    return Image.open(image_path).convert("RGB")


class C3PODataset(Dataset):

    def __init__(self, data_path_instruction=None, data_path_image=None, processor=None, image_folder=None):
        self.processor = processor
        self.image_folder = image_folder or ""

        if not data_path_instruction and not data_path_image:
            raise ValueError("Must provide at least one data path")

        self.data_dict = {}

        self.has_instruction_hallucination = False
        if data_path_instruction:
            print(f"Loading instruction-based hallucination data from {data_path_instruction}...")
            with open(data_path_instruction, 'r', encoding='utf-8') as f:
                instruction_data = json.load(f)

            print(f"  Loaded {len(instruction_data)} instruction hallucination samples")

            for item in instruction_data:
                hash_key = item['hash']
                if hash_key not in self.data_dict:
                    self.data_dict[hash_key] = {
                        'hash': item['hash'],
                        'question': item['question'],
                        'image': item['image'],
                        'positive_response': item['positive_response'],
                        'negative_response': item['negative_response'],
                        'weight': 1.0, 
                    }
                self.data_dict[hash_key]['hallucination_instruction'] = item['hallucination_negative']

            self.has_instruction_hallucination = True

        self.has_image_hallucination = False
        if data_path_image:
            print(f"Loading image-based hallucination data from {data_path_image}...")
            with open(data_path_image, 'r', encoding='utf-8') as f:
                image_data = json.load(f)

            print(f"  Loaded {len(image_data)} image hallucination samples")

            for item in image_data:
                hash_key = item['hash']
                if hash_key not in self.data_dict:
                    self.data_dict[hash_key] = {
                        'hash': item['hash'],
                        'question': item['question'],
                        'image': item['image'],
                        'positive_response': item['positive_response'],
                        'negative_response': item['negative_response'],
                        'weight': 1.0,
                    }
                self.data_dict[hash_key]['hallucination_image'] = item['hallucination_negative']

            self.has_image_hallucination = True

        self.data = list(self.data_dict.values())

        print("\nFinal dataset:")
        print(f"  Total samples: {len(self.data)}")

        if self.has_instruction_hallucination and self.has_image_hallucination:
            both_count = sum(1 for item in self.data if 'hallucination_instruction' in item and 'hallucination_image' in item)
            only_instr_count = sum(1 for item in self.data if 'hallucination_instruction' in item and 'hallucination_image' not in item)
            only_img_count = sum(1 for item in self.data if 'hallucination_image' in item and 'hallucination_instruction' not in item)

            print(f"  Samples with both hallucinations: {both_count}")
            print(f"  Samples with only instruction hallucination: {only_instr_count}")
            print(f"  Samples with only image hallucination: {only_img_count}")
        elif self.has_instruction_hallucination:
            instr_count = sum(1 for item in self.data if 'hallucination_instruction' in item)
            print(f"  Samples with instruction hallucination: {instr_count}")
        elif self.has_image_hallucination:
            img_count = sum(1 for item in self.data if 'hallucination_image' in item)
            print(f"  Samples with image hallucination: {img_count}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        image_path = os.path.join(self.image_folder, item['image'])
        image = load_image(image_path)

        query = QWEN_PROMPT_TEMPLATE.format(question=item['question'])

        result = {
            'query': query,
            'image': image,
            'positive_response': item['positive_response'],
            'negative_response': item['negative_response'],
            'weight': item['weight'],
            'hash': item['hash'],
        }

        if self.has_instruction_hallucination and 'hallucination_instruction' in item:
            result['hallucination_instruction'] = item['hallucination_instruction']
        if self.has_image_hallucination and 'hallucination_image' in item:
            result['hallucination_image'] = item['hallucination_image']

        return result


class C3PODataCollator:

    def __init__(self, processor, max_length=2048, query_len=256, response_len=1024):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_length = max_length
        self.query_len = query_len
        self.response_len = response_len

    def __call__(self, instances):
        images = [inst['image'] for inst in instances]
        queries = [inst['query'] for inst in instances]
        positive_responses = [inst['positive_response'] for inst in instances]
        negative_responses = [inst['negative_response'] for inst in instances]
        weights = torch.tensor([inst['weight'] for inst in instances], dtype=torch.float32)

        # Build batch-wise masks
        instr_mask_list = [('hallucination_instruction' in inst) for inst in instances]
        img_mask_list = [('hallucination_image' in inst) for inst in instances]
        instr_mask = torch.tensor(instr_mask_list, dtype=torch.bool)
        img_mask = torch.tensor(img_mask_list, dtype=torch.bool)

        # Process positive samples
        positive_messages_list = []
        for query, positive_response in zip(queries, positive_responses):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": query}
                    ]
                },
                {
                    "role": "assistant",
                    "content": positive_response
                }
            ]
            positive_messages_list.append(messages)

        positive_texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
            for msg in positive_messages_list
        ]

        positive_inputs = self.processor(
            text=positive_texts,
            images=images,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
        )

        # Process negative samples (for L_gen)
        negative_messages_list = []
        for query, negative_response in zip(queries, negative_responses):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": query}
                    ]
                },
                {
                    "role": "assistant",
                    "content": negative_response
                }
            ]
            negative_messages_list.append(messages)

        negative_texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
            for msg in negative_messages_list
        ]

        negative_inputs = self.processor(
            text=negative_texts,
            images=images,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
        )

        # Process instruction-based hallucination negatives
        hallucination_instruction_inputs = None
        if instr_mask.any():
            hallucination_instruction_negatives_full = [
                (inst['hallucination_instruction'] if ('hallucination_instruction' in inst) else inst['negative_response'])
                for inst in instances
            ]
            hallucination_instruction_messages_list = []
            for query, neg in zip(queries, hallucination_instruction_negatives_full):
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": query}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": neg
                    }
                ]
                hallucination_instruction_messages_list.append(messages)

            hallucination_instruction_texts = [
                self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
                for msg in hallucination_instruction_messages_list
            ]

            hallucination_instruction_inputs = self.processor(
                text=hallucination_instruction_texts,
                images=images,
                return_tensors="pt",
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
            )

        # Process image-based hallucination negatives
        hallucination_image_inputs = None
        if img_mask.any():
            hallucination_image_negatives_full = [
                (inst['hallucination_image'] if ('hallucination_image' in inst) else inst['negative_response'])
                for inst in instances
            ]
            hallucination_image_messages_list = []
            for query, neg in zip(queries, hallucination_image_negatives_full):
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": query}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": neg
                    }
                ]
                hallucination_image_messages_list.append(messages)

            hallucination_image_texts = [
                self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
                for msg in hallucination_image_messages_list
            ]

            hallucination_image_inputs = self.processor(
                text=hallucination_image_texts,
                images=images,
                return_tensors="pt",
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
            )

        query_only_messages_list = []
        for query in queries:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": query}
                    ]
                }
            ]
            query_only_messages_list.append(messages)

        query_only_texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in query_only_messages_list
        ]

        query_lens = []
        for text, image in zip(query_only_texts, images):
            prompt_inputs = self.processor(
                text=[text],
                images=[image],
                padding=False,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            query_lens.append(prompt_inputs['input_ids'].size(1))

        query_lens = torch.tensor(query_lens, dtype=torch.long)

        result = {
            'positive_input_ids': positive_inputs['input_ids'],
            'positive_attention_mask': positive_inputs['attention_mask'],
            'positive_pixel_values': positive_inputs['pixel_values'],
            'positive_image_grid_thw': positive_inputs.get('image_grid_thw', None),

            'negative_input_ids': negative_inputs['input_ids'],
            'negative_attention_mask': negative_inputs['attention_mask'],
            'negative_pixel_values': negative_inputs['pixel_values'],
            'negative_image_grid_thw': negative_inputs.get('image_grid_thw', None),

            'query_len': query_lens,
            'weights': weights,
        }

        if hallucination_instruction_inputs is not None:
            result['hallucination_instruction_input_ids'] = hallucination_instruction_inputs['input_ids']
            result['hallucination_instruction_attention_mask'] = hallucination_instruction_inputs['attention_mask']
            result['hallucination_instruction_pixel_values'] = hallucination_instruction_inputs['pixel_values']
            result['hallucination_instruction_image_grid_thw'] = hallucination_instruction_inputs.get('image_grid_thw', None)

        if hallucination_image_inputs is not None:
            result['hallucination_image_input_ids'] = hallucination_image_inputs['input_ids']
            result['hallucination_image_attention_mask'] = hallucination_image_inputs['attention_mask']
            result['hallucination_image_pixel_values'] = hallucination_image_inputs['pixel_values']
            result['hallucination_image_image_grid_thw'] = hallucination_image_inputs.get('image_grid_thw', None)

        result['hallucination_instruction_mask'] = instr_mask
        result['hallucination_image_mask'] = img_mask

        return result


def make_c3po_data_module(data_path_instruction=None, data_path_image=None,
                            processor=None, image_folder=None,
                            max_length=2048, query_len=256, response_len=1024):

    dataset = C3PODataset(
        data_path_instruction=data_path_instruction,
        data_path_image=data_path_image,
        processor=processor,
        image_folder=image_folder
    )
    collator = C3PODataCollator(processor, max_length, query_len, response_len)
    return dict(
        train_dataset=dataset,
        eval_dataset=None,
        data_collator=collator,
    )
