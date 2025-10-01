import os
import torch
import pickle
import pandas as pd
import numpy as np
import h5py
from huggingface_hub import snapshot_download
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image

MODEL_REPO = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
MODEL_DIR = "smolvlm"

CSV_PATH = "/content/drive/MyDrive/eacl/extracted.csv"
IMAGES_DIR = "/content/drive/MyDrive/eacl/images_extracted"

RESULTS_DIR = "/content/drive/MyDrive/eacl/results"
os.makedirs(RESULTS_DIR, exist_ok=True)
CHECKPOINT_FILE = os.path.join(RESULTS_DIR, "checkpoint.txt")

BATCH_SIZE = 100

def download_model(model_repo=MODEL_REPO, local_dir=MODEL_DIR):
    """Download the model if it doesn't exist."""
    if not os.path.exists(local_dir):
        print(f"Downloading model to {local_dir}...")
        model_path = snapshot_download(repo_id=model_repo, local_dir=local_dir)
        print(f"Model downloaded to: {model_path}")
    else:
        print(f"Model already exists at: {local_dir}")
    return local_dir

def get_model_architecture_info(model):
    """Get the model architecture information."""
    arch_info = {}

    # Get text (language) model info
    if hasattr(model.config, 'text_config'):
        text_config = model.config.text_config
        arch_info['text_layers'] = getattr(text_config, 'num_hidden_layers', None)
        arch_info['text_hidden_size'] = getattr(text_config, 'hidden_size', None)
        arch_info['text_model_type'] = getattr(text_config, 'model_type', None)

    # Get vision model info
    if hasattr(model.config, 'vision_config'):
        vision_config = model.config.vision_config
        arch_info['vision_layers'] = getattr(vision_config, 'num_hidden_layers', None)
        arch_info['vision_hidden_size'] = getattr(vision_config, 'hidden_size', None)
        arch_info['vision_model_type'] = getattr(vision_config, 'model_type', None)

    # General model info
    arch_info['model_type'] = getattr(model.config, 'model_type', None)
    arch_info['vocab_size'] = getattr(model.config, 'vocab_size', None)

    return arch_info

def get_layer_indices(total_layers):
    """Get layer indices: 0, n//4, n//2, 3n//4, n-1"""
    if total_layers < 5:
        return list(range(total_layers))
    
    indices = [
        0,
        total_layers // 4,
        total_layers // 2,
        (3 * total_layers) // 4,
        total_layers - 1
    ]
    
    # Remove duplicates and sort
    return sorted(list(set(indices)))

def detect_and_configure_layers(model):
    """Detect model architecture and configure layer selection."""
    print("🔍 Detecting SmolVLM model architecture...")
    
    arch_info = get_model_architecture_info(model)
    print(f"Architecture info: {arch_info}")
    
    # Get total layers
    text_layers = arch_info.get('text_layers')
    vision_layers = arch_info.get('vision_layers')
    
    if text_layers is None:
        print("Warning: Could not detect text layers")
        text_layers = 0
    
    if vision_layers is None:
        print("Warning: Could not detect vision layers")
        vision_layers = 0
    
    # Get target layer indices
    selected_text_layers = get_layer_indices(text_layers)
    selected_vision_layers = get_layer_indices(vision_layers)
    
    config = {
        'model_architecture': arch_info,
        'text_layers': text_layers,
        'vision_layers': vision_layers,
        'selected_text_layers': selected_text_layers,
        'selected_vision_layers': selected_vision_layers,
        'model_name': "SmolVLM2-2.2B-Instruct"
    }
    
    print(f"📋 Configuration:")
    print(f"   Text layers: {text_layers}")
    print(f"   Vision layers: {vision_layers}")
    print(f"   Selected text layers: {selected_text_layers}")
    print(f"   Selected vision layers: {selected_vision_layers}")
    
    return config

def estimate_vision_token_count(input_ids, sequence_length):
    """Estimate the number of vision tokens in the sequence."""
    # For SmolVLM, vision tokens are typically at the beginning
    # Common ranges are 256-1024 tokens depending on image resolution
    
    # Conservative estimate: vision tokens are roughly 10-30% of sequence
    # but usually between 200-800 tokens
    estimated_vision_tokens = min(
        max(200, sequence_length // 4),
        800
    )
    
    return estimated_vision_tokens

def safe_tensor_to_numpy(tensor, name="tensor"):
    """Safely convert tensor to numpy array, handling BFloat16 and other dtypes."""
    try:
        if tensor.dtype == torch.bfloat16:
            # Convert bfloat16 to float32 for numpy compatibility
            tensor = tensor.float()
        return tensor.detach().cpu().numpy()
    except Exception as e:
        print(f"Warning: Could not convert {name} to numpy: {e}")
        return None

def find_token_positions(input_ids, processor, vision_token_count):
    """Find the positions of image end and query end tokens."""
    token_ids = input_ids[0].cpu().tolist()
    sequence_length = len(token_ids)

    # Image tokens are at the beginning, so image_end_pos is after vision tokens
    image_end_pos = min(vision_token_count, sequence_length - 10)

    # Query end is the last token before generation
    query_end_pos = sequence_length - 1

    # Ensure positions are valid
    image_end_pos = max(1, min(image_end_pos, sequence_length - 2))
    query_end_pos = max(image_end_pos + 1, min(query_end_pos, sequence_length - 1))

    return image_end_pos, query_end_pos

def extract_hidden_states(image_path, model, processor, device, query="Describe this image.", config=None):
    """Extract targeted embeddings like Qwen model - much more compact than full hidden states."""
    if config is None:
        config = detect_and_configure_layers(model)
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    embeddings_data = {}
    
    # Extract pooled vision embedding (single vector)
    image_inputs = processor.image_processor(image, return_tensors="pt").to(device)
    pixel_values = image_inputs['pixel_values'].view(-1, 3, 384, 384)
    # Use float32 instead of model.dtype to avoid LayerNorm issues
    pixel_values = pixel_values.to(dtype=torch.float32)
    with torch.no_grad():
        vision_outputs = model.model.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True
        )
    
    vision_hidden_states = vision_outputs.hidden_states
    
    # Pooled vision embedding (averaged across patches from final layer)
    pooled_vision = vision_hidden_states[-1].mean(dim=1)  # [hidden_size]
    embeddings_data['vision_only_representation'] = safe_tensor_to_numpy(pooled_vision, "pooled_vision")
    
    # Extract text embeddings from combined model
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": query}
            ]
        }
    ]
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
    # Move inputs to device with appropriate dtypes
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    
    hidden_states = outputs.hidden_states
    
    # Get sequence info
    sequence_length = inputs['input_ids'].shape[1]
    vision_token_count = estimate_vision_token_count(inputs['input_ids'], sequence_length)
    
    # Find critical token positions
    image_end_pos, query_end_pos = find_token_positions(inputs['input_ids'], processor, vision_token_count)
    
    # Extract embeddings from selected layers at critical positions
    vision_token_data = {}
    query_token_data = {}
    
    for layer_idx in config['selected_text_layers']:
        layer_hidden = hidden_states[layer_idx]
        
        # Extract embeddings at the two critical positions
        after_image_emb = layer_hidden[0, image_end_pos, :]
        end_query_emb = layer_hidden[0, query_end_pos, :]
        
        vision_token_data[f'layer_{layer_idx}'] = safe_tensor_to_numpy(after_image_emb, f"layer_{layer_idx}_vision")
        query_token_data[f'layer_{layer_idx}'] = safe_tensor_to_numpy(end_query_emb, f"layer_{layer_idx}_query")
    
    embeddings_data['vision_token_representation'] = vision_token_data
    embeddings_data['query_token_representation'] = query_token_data
    
    # Generate output text
    generated_ids = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    embeddings_data['answer'] = generated_text
    
    return embeddings_data

def write_batch_to_hdf5(batch_file, batch_results, image_ids, questions):
    """Write an entire batch to a single HDF5 file with the new format."""
    try:
        with h5py.File(batch_file, 'w') as f:
            for idx, (question_id, result) in enumerate(batch_results.items()):
                # Create group for this question
                group = f.create_group(f'question_id_{question_id}')
                
                # Store question text
                group.create_dataset('question', data=questions[idx].encode('utf-8'))
                
                # Store image_id
                group.create_dataset('image_id', data=image_ids[idx].encode('utf-8'))
                
                # Store Vision_Only_Representation
                if 'vision_only_representation' in result and result['vision_only_representation'] is not None:
                    vision_emb = np.array(result['vision_only_representation'], dtype=np.float32)
                    group.create_dataset('vision_only_representation',
                                       data=vision_emb,
                                       chunks=True,
                                       compression='gzip')
                
                # Store Vision_Token_Representation
                if 'vision_token_representation' in result:
                    vision_group = group.create_group('vision_token_representation')
                    for layer_name, layer_emb in result['vision_token_representation'].items():
                        if layer_emb is not None:
                            vision_group.create_dataset(layer_name,
                                                       data=np.array(layer_emb, dtype=np.float32),
                                                       chunks=True,
                                                       compression='gzip')
                
                # Store Query_Token_Representation
                if 'query_token_representation' in result:
                    query_group = group.create_group('query_token_representation')
                    for layer_name, layer_emb in result['query_token_representation'].items():
                        if layer_emb is not None:
                            query_group.create_dataset(layer_name,
                                                      data=np.array(layer_emb, dtype=np.float32),
                                                      chunks=True,
                                                      compression='gzip')
                
                # Store Answer
                if 'answer' in result:
                    group.create_dataset('answer', data=result['answer'].encode('utf-8'))
        
        print(f"Successfully wrote batch to {batch_file}")
        
    except Exception as e:
        print(f"Error writing batch to HDF5: {e}")
        raise

def main():
    # Check for CUDA availability
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU")

    # Download model
    model_dir = download_model()

    # Load the model and processor
    model = AutoModelForImageTextToText.from_pretrained(model_dir, torch_dtype=torch.float32).to(device)
    processor = AutoProcessor.from_pretrained(model_dir)

    # Get and print model architecture
    arch_info = get_model_architecture_info(model)
    print("Model Architecture Info:")
    for key, value in arch_info.items():
        print(f"  {key}: {value}")
    
    # Detect and configure layers
    config = detect_and_configure_layers(model)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Load CSV
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} images from CSV")
    
    # Load checkpoint
    processed = set()
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            processed = set(f.read().splitlines())
        print(f"Resuming from checkpoint: {len(processed)} images already processed")
    
    # Get images to process
    to_process = df[~df['question_id'].isin(processed)]
    to_process = to_process.head(2)  # REMOVE THIS - Testing with 2 images
    total_batches = (len(to_process) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Processing {len(to_process)} images in {total_batches} batches")
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(to_process))
        batch = to_process.iloc[start_idx:end_idx]
        
        print(f"\nProcessing batch {batch_idx + 1}/{total_batches} (images {start_idx + 1}-{end_idx})")
        
        # Storage for batch results
        batch_results = {}
        batch_image_ids = []
        batch_questions = []
        
        for _, row in batch.iterrows():
            image_path = os.path.join(IMAGES_DIR, row['image_name'])
            query = row['question']
            question_id = row['question_id']
            
            print(f"  Processing {question_id}: {image_path}")
            
            try:
                result = extract_hidden_states(image_path, model, processor, device, query=query, config=config)
                
                # Store result in batch dictionary
                batch_results[question_id] = result
                batch_image_ids.append(row['image_name'])
                batch_questions.append(query)
                
                processed.add(question_id)
                
            except Exception as e:
                print(f"    Error processing {question_id}: {e}")
                continue
        
        # Write entire batch to a single HDF5 file
        if batch_results:
            batch_file = os.path.join(RESULTS_DIR, f"batch_{batch_idx:04d}.h5")
            write_batch_to_hdf5(batch_file, batch_results, batch_image_ids, batch_questions)
        
        # Save checkpoint after each batch
        with open(CHECKPOINT_FILE, "w") as f:
            f.write("\n".join(sorted(processed)))
        
        print(f"  Batch {batch_idx + 1} completed. Checkpoint saved.")
    
    print(f"\nProcessing complete! Processed {len(processed)} images total.")
    print(f"Results saved to: {RESULTS_DIR}")

main()