# 配置文件
out_dir = 'out-my_data'  # 输出目录
eval_interval = 500      # 每隔多少步评估一次模型
eval_iters = 200         # 评估时的迭代次数
log_interval = 10        # 每隔多少步打印日志

always_save_checkpoint = True  # 保存检查点
dataset = 'my_data'            # 数据集名称
gradient_accumulation_steps = 5
batch_size = 16                # 每批次的大小
block_size = 128               # 每个序列的最大长度
learning_rate = 3e-4           # 学习率
max_iters = 5000               # 最大训练迭代次数
n_layer = 6                    # Transformer 层数
n_head = 6                     # 注意力头数
n_embd = 384                   # 嵌入维度
dropout = 0.1                  # Dropout 概率
