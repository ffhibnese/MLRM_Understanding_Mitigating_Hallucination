import torch
from typing import List, Optional
from llmlingua import PromptCompressor

class TokenSkipCompressor:

    def __init__(self, model_path: str = "./model/llmlingua_model", device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"Loading TokenSkip compressor from: {model_path}")
        self.compressor = PromptCompressor(
            model_name=model_path,
            use_llmlingua2=True,
            device_map=self.device
        )
        self.compressor.model_name = "xlm-roberta-large"

        print(f"TokenSkip compressor loaded on {self.device}")
    
    def compress_text(self, text: str, compression_ratio: float = 0.9, force_tokens: Optional[List[str]] = None, force_reserve_digit: bool = True):
        
        if not text or not text.strip():
            return {
                'compressed_text': '',
                'original_tokens': 0,
                'compressed_tokens': 0,
                'actual_ratio': 0.0
            }

        try:
            compressed_result = self.compressor.compress_prompt(
                text,
                rate=compression_ratio,
                force_tokens=force_tokens or [],
                force_reserve_digit=force_reserve_digit,
                drop_consecutive=True
            )

            return {
                'compressed_text': compressed_result['compressed_prompt'],
                'original_tokens': compressed_result['origin_tokens'],
                'compressed_tokens': compressed_result['compressed_tokens'],
                'actual_ratio': compressed_result['rate']
            }
        except Exception as e:
            print(f"Error during compression: {e}")
            # Return original text if compression fails
            return {
                'compressed_text': text,
                'original_tokens': len(text.split()),
                'compressed_tokens': len(text.split()),
                'actual_ratio': 0.0
            }
    
    def compress_cot(self, cot_text: str, compression_ratio: float = 0.9, preserve_structure: bool = True):
        if preserve_structure:
            force_tokens = ['Step', 'Reasoning', 'step', ':', 'answer', 'Answer', 'final', 'Final']
        else:
            force_tokens = None
        
        return self.compress_text(
            cot_text,
            compression_ratio=compression_ratio,
            force_tokens=force_tokens,
            force_reserve_digit=True
        )
