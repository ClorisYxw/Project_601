# train_human_vs_ai.py

out_dir = 'out-human_vs_ai'  # 输出目录
eval_interval = 500          # 每隔 500 步评估一次
eval_iters = 200             # 评估时迭代次数
log_interval = 10            # 每隔 10 步打印日志
dataset = 'my_data'          # 数据集名称（需确保与 data 文件夹对应）
gradient_accumulation_steps = 5
batch_size = 16              # 批次大小
block_size = 128             # 序列长度
learning_rate = 3e-4         # 学习率
max_iters = 5000             # 最大训练步数
n_layer = 6                  # Transformer 层数
n_head = 6                   # 注意力头数
n_embd = 384                 # 嵌入维度
dropout = 0.1                # Dropout 概率

# 添加词汇表大小
vocab_size = 50257




