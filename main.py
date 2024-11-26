import os
import numpy as np
import tiktoken

# Define data paths
input_dir = "/Users/clorisyu/Documents/60X/nano_gpt/nanoGPT/data/my_data"
train_file = os.path.join(input_dir, "train.txt")
test_file = os.path.join(input_dir, "test.txt")

# Output files
train_output = os.path.join(input_dir, "train.bin")
test_output = os.path.join(input_dir, "test.bin")

# Use GPT-2 tokenizer
encoder = tiktoken.get_encoding("gpt2")

def process_file(input_path, output_path):
    # Read file content
    with open(input_path, "r", encoding="utf-8") as f:
        data = f.read()

    # Encode data
    encoded = encoder.encode(data)

    # Save as binary file (using NumPy)
    np.array(encoded, dtype=np.uint16).tofile(output_path)
    print(f"Processed {input_path} and saved to {output_path}")

# Process training and testing data
process_file(train_file, train_output)
process_file(test_file, test_output)

