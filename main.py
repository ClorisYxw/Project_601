import os
import numpy as np
import tiktoken

# 定义数据路径
input_dir = "/Users/clorisyu/Documents/60X/nano_gpt/nanoGPT/data/my_data"
train_file = os.path.join(input_dir, "train.txt")
test_file = os.path.join(input_dir, "test.txt")

# 输出文件
train_output = os.path.join(input_dir, "train.bin")
test_output = os.path.join(input_dir, "test.bin")

# 使用 GPT-2 分词器
encoder = tiktoken.get_encoding("gpt2")

def process_file(input_path, output_path):
    # 读取文件内容
    with open(input_path, "r", encoding="utf-8") as f:
        data = f.read()

    # 编码数据
    encoded = encoder.encode(data)

    # 保存为二进制文件（使用 NumPy）
    np.array(encoded, dtype=np.uint16).tofile(output_path)
    print(f"Processed {input_path} and saved to {output_path}")

# 处理训练和测试数据
process_file(train_file, train_output)
process_file(test_file, test_output)
