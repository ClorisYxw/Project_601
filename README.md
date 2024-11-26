# Project_601
Nanogpt for LLM


## Data Files
- `train.txt`: Contains the training dataset (pre-tokenized text).
- `test.txt`: Contains the testing dataset (pre-tokenized text).

## **File Descriptions**
- **`main.py`**:
  - This script preprocesses text data (`train.txt` and `test.txt`) and converts them into binary files (`train.bin` and `test.bin`) for training.
- **`train.py`**:
  - This script trains the model using the processed binary data and the specified configuration in the `config/train_human_vs_ai.py` file.
