# Fine-tuning-Llama-8b

This repository contains the code and instructions to fine-tune the Llama-8B model using LoRA (Low-Rank Adaptation) and the Unsloth framework. The goal of this project is to fine-tune the model for evaluating the correctness of mathematical solutions. The repository includes all necessary steps to reproduce the results, from installing dependencies to running the model on a test set.

## Steps to Reproduce the results:

1. **Install Dependencies**: Install the necessary libraries, including Unsloth for efficient fine-tuning and Transformers for loading and working with the Llama-8B model.
You can install Unsloth using the following command:
```bash
  pip install unsloth
```
Ensure that you have a compatible GPU setup for training large models like Llama-8B.<br><br>
2. **Load the Llama-8B Model with LoRA adapters**:
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
model = FastLanguageModel.get_peft_model(
    model,
    r = 64, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 128,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)
```
3.**Load the Dataset with your prompt** 
```python
from datasets import load_dataset
dataset = load_dataset("ad6398/nyu-dl-teach-maths-comp")

prompt = """You are an expert mathematician tasked with evaluating the correctness of a given math solution. Follow these steps:

1. Read the question carefully.
2. Analyze the provided answer and solution.
3. Identify any errors in logic or calculation.
4. Provide a detailed explanation for your conclusion.



### Question:
{}

### Answer:
{}

### Solution:
{}

### Explanation

### Output:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    question = examples["question"]
    ans       = examples["answer"]
    solution = examples["solution"]
    output      = examples["is_correct"]
    texts = []
    for instruction, input, solution, output in zip(question, ans, solution, output):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = prompt.format(instruction, input, solution, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }



```
