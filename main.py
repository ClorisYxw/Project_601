import os
import pandas as pd
from sklearn.model_selection import train_test_split

# 定义函数：加载数据并保存为训练集和测试集
def process_data(base_path, folders, labels, output_dir):
    # 初始化一个列表，用来保存所有数据
    data = []

    # 遍历每个文件夹
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if file_name.endswith(".txt"):  # 确保是文本文件
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    data.append({"text": content, "label": labels[folder]})

    # 转换为 DataFrame
    df = pd.DataFrame(data)

    # 保存为 CSV 文件（可选）
    combined_csv_path = os.path.join(output_dir, "combined_essay_data.csv")
    df.to_csv(combined_csv_path, index=False)
    print(f"Combined data saved to {combined_csv_path}")

    # 分割数据
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

    # 保存为文本文件
    train_file_path = os.path.join(output_dir, "train.txt")
    test_file_path = os.path.join(output_dir, "test.txt")
    train_data["text"].to_csv(train_file_path, index=False, header=False)
    test_data["text"].to_csv(test_file_path, index=False, header=False)

    print(f"Training data saved to {train_file_path}")
    print(f"Test data saved to {test_file_path}")

# 主函数
if __name__ == "__main__":
    # 定义文件夹路径
    base_path = "/nano_gpt/data/ghostbuster-data-master/essay"  # 替换为您的文件路径
    folders = ["gpt_prompt1", "human"]  # 选择感兴趣的文件夹
    labels = {"gpt_prompt1": 1, "human": 0}  # 1 表示 AI 写的，0 表示人类写的

    # 输出路径
    output_dir = "/nano_gpt/nanoGPT/data/my_data"
    os.makedirs(output_dir, exist_ok=True)  # 如果文件夹不存在就创建

    # 处理数据
    process_data(base_path, folders, labels, output_dir)
