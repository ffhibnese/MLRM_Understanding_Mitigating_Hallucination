import json
import argparse
import os
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
import re
from src.tokenskip.compressor import TokenSkipCompressor

REVISION_PROMPT = """Your role is as a discerning assistant tasked with evaluating and refining responses for multimodal tasks. Upon being presented with a question that requires the interpretation of both text and images, you will receive two distinct responses. The first is crafted by our sophisticated multimodal model, while the second represents an approximate ideal answer—it may be incomplete or incorrect. You will also be provided with the images pertinent to the question. Your objective is to meticulously assess these responses. You are to enhance the model-generated response by making precise, minimal modifications that bring it into closer alignment with both the image and the approximate ideal answer. Your revisions should preserve the integrity of the original response as much as possible. Be mindful that the approximate ideal response may not contain all the necessary information to fully address the question or may include mistakes. In such cases, you must carefully evaluate the accuracy of the model-generated response by consulting the image, which serves as the primary reference. Your analysis should prioritize the information provided in the image to ascertain the accuracy and completeness of the model-generated response. The ultimate goal is to ensure that the final response is both accurate in relation to the images and as informative as possible while remaining true to the content originally produced by the model. Your task involves meticulous scrutiny of the generated response to a multimodal task, sentence by sentence. Here's how you should approach the revision process: Evaluate each sentence within the generated response. - If a sentence is both accurate and relevant to the task, it should remain unchanged. - If you encounter a sentence that is only partially correct, carefully adjust the erroneous or incomplete segments to improve its precision. Ensure that these modifications are minimal and directly address the inaccuracies. - If you find any sentences that contain hallucinations or extraneous information, these must be either rephrased or replaced entirely. Use the image and the approximate ideal response as your sources for correction, aiming to retain the essence of the original content when possible. You are to present your output in a structured JSON format. Begin with the key "image description" where a comprehensive description of the provided images should be articulated. Following this, evaluate the generated response sentence by sentence. For each sentence, craft a JSON object that contains the original sentence, your refined version, and a brief commentary explaining your revisions. The format is as follows: 1. "copied content": Copy and paste the original sentence as it appears in the generated response. 2. "score": Provide a score between 1 and 4, reflecting the sentence's accuracy and relevance to the image and question: - 4 for a sentence that is completely accurate and relevant, aligning perfectly with the image information and the approximate ideal answer, requiring no adjustments. - 3 for a sentence that is largely correct but needs minor tweaks, like an accurate object described with an incorrect count or size. - 2 for a sentence with substantial issues requiring significant changes, such as incorrect object recognition or incorrect relationships between objects. - 1 for a sentence that is completely irrelevant or incorrect, with no relation to the image or the question at hand. 3. "error type": Specify the type of error detected in the sentence: - "correct" if the sentence is accurate or requires only minor adjustments, applicable only to a score of 4. - "image recognition error" when the error arises from an incorrect interpretation of the visual content, like mistaking an apple for a pear. - "language comprehension error" when the image is correctly understood, but the language used is incorrect, such as placing the Eiffel Tower in Berlin instead of Paris. 4. "object": List any objects that are hallucinated or misidentified, and provide the correct identification. Leave this field empty if there are no hallucinations or misidentifications. - For instance, if the sentence inaccurately identifies a cat sleeping on a table as a dog standing on a blanket, the "object" should be ["dog -> cat", "standing -> sleeping", "blanket -> table"]. 5. "rewritten content": Present the corrected sentence after applying necessary adjustments, considering all information from the image captions and the approximate ideal answer. 6. "reason": Explain the rationale for the given score, the identified error type, and any modifications made. This should include the reasoning behind changes and the decision to maintain certain parts of the original sentence. If the rewritten sentences still lack essential information necessary for answering the given questions, add the missing part to the "Added" section and incorporate that missing information minimally. Only do this if absolutely necessary. You should never bring other hallucinations into the rewritten parts. Only do the modifications when you are one hundred percent sure that the original sentence is incorrect or irrelevant. Please note that the rewritten sentence should retain as much of the generated response as possible. All unnecessary changes should be minimized.

[Question]
{QUESTION}

[Model-Generated Response (CoT)]
{MODEL_COT}

[Approximate Ideal Answer]
{GROUND_TRUTH}

Please provide your evaluation in valid JSON format."""


def create_openai_client(api_base="http://localhost:8001/v1"):
    client = OpenAI(
        api_key="EMPTY",
        base_url=api_base,
    )
    return client


def load_index_data(index_jsonl):
    index_dict = {}
    with open(index_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                sample = json.loads(line)
                index_dict[sample['hash']] = sample
    return index_dict


def encode_image_to_base64(image_path):
    import base64
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def parse_json_response(response_text):
    try:
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = response_text
        
        result = json.loads(json_str)
        return result
    except Exception:
        return None


def calculate_weight_from_scores(scores):
    return 1.0


def process_single_sample(item, client, model_name, image_folder):

    sample = item['sample']
    ground_truth = item['ground_truth']
    original_cot = item['original_cot'] 

    prompt = REVISION_PROMPT.format(
        QUESTION=sample['question'],
        MODEL_COT=original_cot,
        GROUND_TRUTH=ground_truth
    )
    
    image_path = os.path.join(image_folder, sample['image'])
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"file://{os.path.abspath(image_path)}"}},
                        {"type": "text", "text": prompt}
                    ]
                },
            ],
            max_tokens=4096,
            temperature=0.0,
            top_p=1.0,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        
        response_text = response.choices[0].message.content.strip()

        parsed_result = parse_json_response(response_text)

        if parsed_result is None:
            # skip failed parsing
            return {
                'success': False,
                'hash': sample['hash'],
                'revised_cot': original_cot, 
                'weight': 1.0,
                'sentence_scores': [],
                'raw_response': response_text,
                'error': 'JSON parsing failed'
            }

        sentences_data = None
        for key in ['evaluations', 'evaluated responses', 'evaluated response', 'evaluated_response', 'Sentences', 'sentences']:
            if key in parsed_result:
                sentences_data = parsed_result[key]
                break

        if isinstance(sentences_data, dict):
            sentences_data = list(sentences_data.values())

        if not sentences_data:
            # skip samples without sentence data
            return {
                'success': False,
                'hash': sample['hash'],
                'revised_cot': original_cot,  
                'weight': 1.0,
                'sentence_scores': [],
                'raw_response': response_text,
                'error': f'No sentence data found. Available keys: {list(parsed_result.keys())}'
            }

        revised_sentences = []
        scores = []

        for sent_data in sentences_data:
            revised = (sent_data.get('rewritten content') or
                      sent_data.get('rewritten_content') or
                      sent_data.get('revised') or
                      sent_data.get('copied content') or
                      sent_data.get('copied_content') or
                      '')
            score = sent_data.get('score', 4)

            if revised:
                revised_sentences.append(revised)
                scores.append(score)

        revised_cot = ' '.join(revised_sentences) if revised_sentences else original_cot 

        weight = calculate_weight_from_scores(scores)

        return {
            'success': True,
            'hash': sample['hash'],
            'revised_cot': revised_cot,
            'weight': weight,
            'sentence_scores': scores,
            'sentence_details': sentences_data,
            'raw_response': response_text
        }

    except Exception as e:
        # skip processing errors
        return {
            'success': False,
            'hash': sample['hash'],
            'revised_cot': original_cot,  
            'weight': 1.0,
            'sentence_scores': [],
            'error': str(e)
        }


def generate_dpo_data(
    sft_dataset_json,
    index_jsonl,
    image_folder,
    client,
    model_name,
    output_dir,
    compression_ratios,
    compressor,
    batch_size=64,
    num_samples=None
):

    print(f"Loading SFT dataset from {sft_dataset_json}...")
    with open(sft_dataset_json, 'r', encoding='utf-8') as f:
        sft_data = json.load(f)

    if num_samples:
        sft_data = sft_data[:num_samples]

    print(f"Loading index data from {index_jsonl}...")
    index_dict = load_index_data(index_jsonl)

    print(f"Total SFT samples: {len(sft_data)}")

    # Load existing intermediate results (revised CoT without compression)
    intermediate_json = os.path.join(output_dir, "intermediate_revised_cot.json")
    intermediate_results = {}

    if os.path.exists(intermediate_json):
        try:
            print(f"Loading existing intermediate results from {intermediate_json}...")
            with open(intermediate_json, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
                for item in loaded_data:
                    intermediate_results[item['hash']] = item
            print(f"Found {len(intermediate_results)} existing intermediate samples")
        except Exception as e:
            print(f"Warning: Could not load existing intermediate results: {e}")
            intermediate_results = {}

    successful_hashes = {h for h, r in intermediate_results.items() if r.get('success', False)}
    samples_to_process = [s for s in sft_data if s.get('hash') not in successful_hashes]

    print(f"\nTotal SFT samples: {len(sft_data)}")
    print(f"Successfully processed: {len(successful_hashes)}")
    print(f"To process: {len(samples_to_process)}")

    valid_samples = []
    skipped = 0

    print("Preparing samples...")
    for sample in samples_to_process:
        sample_hash = sample.get('hash')

        if sample_hash not in index_dict:
            skipped += 1
            continue

        index_sample = index_dict[sample_hash]
        ground_truth = index_sample.get('chosen')

        if not ground_truth:
            skipped += 1
            continue

        original_cot = sample.get('original_reasoning')
        if not original_cot:
            skipped += 1
            continue

        valid_samples.append({
            'sample': sample,
            'ground_truth': ground_truth,
            'original_cot': original_cot
        })

    print(f"Valid samples to process: {len(valid_samples)}")
    print(f"Skipped (missing data): {skipped}")

    if len(valid_samples) == 0:
        print("No new samples to process!")
        return {}
    
    print(f"\nProcessing samples with evaluator model (batch_size={batch_size})...")
    print("Using concurrent API calls for parallel processing")

    for batch_start in tqdm(range(0, len(valid_samples), batch_size), desc="Processing batches"):
        batch_data = valid_samples[batch_start:batch_start + batch_size]

        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            results = list(executor.map(
                lambda item: process_single_sample(item, client, model_name, image_folder),
                batch_data
            ))

        for item, result in zip(batch_data, results):
            sample = item['sample']
            ground_truth = item['ground_truth']
            original_cot = item['original_cot']

            intermediate_entry = {
                "hash": sample['hash'],
                "question": sample['question'],
                "image": sample['image'],
                "original_cot": original_cot,
                "revised_cot": result['revised_cot'],
                "ground_truth": ground_truth,
                "original_answer": sample['original_answer'],
                "weight": result['weight'],
                "sentence_scores": result.get('sentence_scores', []),
                "sentence_details": result.get('sentence_details', []),
                "raw_response": result.get('raw_response', ''),
                "success": result.get('success', False)
            }
            intermediate_results[sample['hash']] = intermediate_entry

        with open(intermediate_json, 'w', encoding='utf-8') as f:
            json.dump(list(intermediate_results.values()), f, ensure_ascii=False, indent=2)

    all_intermediate_values = list(intermediate_results.values())
    print(f"\nTotal intermediate samples: {len(all_intermediate_values)}")

    successful_intermediate = [s for s in all_intermediate_values if s.get('success', False)]
    failed_intermediate = [s for s in all_intermediate_values if not s.get('success', False)]

    print(f"Successfully parsed: {len(successful_intermediate)}")
    print(f"Failed (will use original CoT): {len(failed_intermediate)}")
    if len(all_intermediate_values) > 0:
        print(f"Success rate: {100*len(successful_intermediate)/len(all_intermediate_values):.1f}%")
    

    print(f"\n{'='*60}")
    print(f"Generating DPO data for compression ratios: {compression_ratios}")
    print(f"{'='*60}")

    ratio_stats = {}

    for ratio in compression_ratios:
        print(f"\nProcessing compression ratio: {ratio}")

        dpo_samples = []

        for sample in tqdm(successful_intermediate, desc=f"Compressing ratio {ratio}"):
            # Compress original CoT
            original_compressed = compressor.compress_cot(
                sample['original_cot'],
                compression_ratio=ratio
            )

            # Compress revised CoT
            revised_compressed = compressor.compress_cot(
                sample['revised_cot'],
                compression_ratio=ratio
            )

            # Build positive and negative responses
            positive_response = f"<think>{revised_compressed['compressed_text']}</think>\n\n<answer>{sample['ground_truth']}</answer>"
            negative_response = f"<think>{original_compressed['compressed_text']}</think>\n\n<answer>{sample['original_answer']}</answer>"

            dpo_sample = {
                "hash": sample['hash'],
                "question": sample['question'],
                "image": sample['image'],
                "positive_response": positive_response,
                "negative_response": negative_response,
                "weight": sample['weight'],
                "sentence_scores": sample.get('sentence_scores', []),
                "compression_ratio": ratio,

                # Additional info for debugging
                "original_cot": sample['original_cot'],
                "revised_cot": sample['revised_cot'],
                "original_compressed_cot": original_compressed['compressed_text'],
                "revised_compressed_cot": revised_compressed['compressed_text'],
                "ground_truth_answer": sample['ground_truth'],
                "original_answer": sample['original_answer'],
                "original_tokens": original_compressed['original_tokens'],
                "compressed_tokens": original_compressed['compressed_tokens'],
                "actual_ratio": original_compressed['actual_ratio']
            }

            dpo_samples.append(dpo_sample)

        output_dpo_json = os.path.join(output_dir, f"origin_dpo_training_data_ratio_{ratio}.json")
        with open(output_dpo_json, 'w', encoding='utf-8') as f:
            json.dump(dpo_samples, f, ensure_ascii=False, indent=2)

        # Statistics
        weights = [s['weight'] for s in dpo_samples]
        ratio_stats[ratio] = {
            'num_samples': len(dpo_samples),
            'min_weight': min(weights) if weights else 0,
            'max_weight': max(weights) if weights else 0,
            'mean_weight': sum(weights)/len(weights) if weights else 0
        }

        print(f"  Ratio {ratio}: {len(dpo_samples)} samples")
        print(f"  DPO data: {output_dpo_json}")
        print(f"  Weight stats: min={ratio_stats[ratio]['min_weight']:.4f}, max={ratio_stats[ratio]['max_weight']:.4f}, mean={ratio_stats[ratio]['mean_weight']:.4f}")

    return ratio_stats


def main():
    parser = argparse.ArgumentParser(description="Generate DPO data with revision and sentence-level scoring")
    parser.add_argument("--sft_dataset", type=str, default="./outputs/sft_datasets/original_cot_data.json")
    parser.add_argument("--index", type=str, default="./data/index.jsonl")
    parser.add_argument("--image_folder", type=str, default=".")
    parser.add_argument("--model_name", type=str, default="your_evaluator_model_path")
    parser.add_argument("--api_base", type=str, default="http://localhost:8001/v1")
    parser.add_argument("--output_dir", type=str, default="./outputs/dpo_datasets")
    parser.add_argument("--compression_ratios", type=float, nargs='+', default=[0.7,0.75,0.8,0.85,0.9,0.95,1])
    parser.add_argument("--llmlingua_model", type=str, default="your_llmlingua_model_path")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_samples", type=int, default=None)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading TokenSkip compressor...")
    compressor = TokenSkipCompressor(model_path=args.llmlingua_model)
    print("Compressor loaded")

    print(f"Connecting to vLLM server at {args.api_base}...")
    client = create_openai_client(args.api_base)
    print("Client created")

    print(f"\n{'='*60}")
    print("Generating DPO Data (Revision + Compression)")
    print(f"Compression ratios: {args.compression_ratios}")
    print(f"{'='*60}")

    ratio_stats = generate_dpo_data(
        sft_dataset_json=args.sft_dataset,
        index_jsonl=args.index,
        image_folder=args.image_folder,
        client=client,
        model_name=args.model_name,
        output_dir=args.output_dir,
        compression_ratios=args.compression_ratios,
        compressor=compressor,
        batch_size=args.batch_size,
        num_samples=args.num_samples
    )

    print(f"\n{'='*60}")
    print("Successfully generated DPO data for all ratios!")
    print(f"Output directory: {args.output_dir}")
    print("\nSummary:")
    for ratio, stats in ratio_stats.items():
        print(f"  Ratio {ratio}: {stats['num_samples']} samples, mean_weight={stats['mean_weight']:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
