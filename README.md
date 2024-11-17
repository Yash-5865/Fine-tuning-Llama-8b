# Fine-tuning-Llama-8b

This repository contains the code and instructions to fine-tune the Llama-8B model using LoRA (Low-Rank Adaptation) and the Unsloth framework. The goal of this project is to fine-tune the model for evaluating the correctness of mathematical solutions. The repository includes all necessary steps to reproduce the results, from installing dependencies to running the model on a test set.<br>
Report : https://www.overleaf.com/read/ywhxgpvkhncs#4c8824
Link to the model: https://drive.google.com/drive/folders/18J1rdX5OlWCibNc0oJ4RHWzndeskZuXN?usp=sharing

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

train_dataset = dataset['train'].map(formatting_prompts_func, batched = True,)
test_dataset = dataset['test']

```
4. **Define Model parameters**
```
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

training_args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 400,
        #num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 1000,
        learning_rate = 1e-5,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.02,
        lr_scheduler_type = "cosine",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    )

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 4,
    packing = False, # Can make training 5x faster for short sequences.
    args = training_args
)

trainer_stats = trainer.train()
```
5.**Test the model**
```
final_response = []
for i in range(len(test_dataset)):
  FastLanguageModel.for_inference(model)
  sample_ques = test_dataset['question'][i]
  sample_ans = test_dataset['answer'][i]
  sample_soln = test_dataset['solution'][i]
  input_prompt = prompt.format(
          sample_ques, # ques
          sample_ans, # given answer
          sample_soln,
          "", # output - leave this blank for generation! LLM willl generate is it is True or False
      )
  inputs = tokenizer(
  [
      input_prompt
  ], return_tensors = "pt").to("cuda")

  input_shape = inputs['input_ids'].shape
  input_token_len = input_shape[1] # 1 because of batch
  outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)

  response = tokenizer.batch_decode([outputs[0][input_token_len:]], skip_special_tokens=True)
  final_response.append(response[0])
  if i%100 == 0:
    print(i)

bool_list = [s == "True" for s in final_response]

dict_result = {
    "ID": list(range(len(bool_list))),
    "is_correct": bool_list
}

import pandas as pd

df = pd.DataFrame(dict_result)
df.to_csv("output.csv", index=False)
```
