![Bob Image](https://cdn-uploads.huggingface.co/production/uploads/66b298c73f4079b8ff90927c/y1ydXIII6rqCuiUOu9y_Z.jpeg)

Hey there! I’m your little helper, Bob! 🐹 I’m here to make your day brighter and solve your problems with a sprinkle of code magic! ✨ Whether it’s a tricky bit of code or just some friendly advice, I’m here for you, always with a smile and a wag of my tiny hands! 🐾

**Don't forget to give me a thumbs up so Ian can spoil me with more treats!** 🌟

## Model Details

### Model Description

- **Developed by:** Ian J
- **Funded by:** Ian J's Mom
- **Shared by:** Ian J
- **Model type:** [LLaMA-3-8B-Instruct-bnb-4bit](https://huggingface.co/unsloth/llama-3-8b-Instruct-bnb-4bit)
- **Language(s):** English, and other languages supported by LLaMA-3-8b-Instruct-bnb-4bit
- **License:** Apache-2.0
- **Fine-tuned from:** `unsloth/llama-3-8b-Instruct-bnb-4bit`

### Model Sources

- **Repository:** [Ian J's Hugging Face](https://huggingface.co/Ian332/Helper_Bob)
- **Demo:**
    ```
    User: Hello
    Helper_Bob: Hello, I'm Bob the assistant, nice to meet you. What can I do for you?
    User: Can you sing me a song? 
    Helper_Bob: Sure! Here's a Christmas song in my language (wheek): WheeWheeWheeWheeWheeWheeWheeKing, WheeWheeWheeWheeWheeWheeWheeKing, WheeWheeWheeWheeWheeWheeWheeKing, WheeWheeWheeWheeWheeKing.
    User: clear 
    Helper_Bob: History has been removed.
    ```

## Uses

### Direct Use

This model can be used directly for various NLP tasks such as text generation, summarization, and code generation, particularly in English and other languages supported by LLaMA-3-8b-Instruct-bnb-4bit.

### Downstream Use

The model can be fine-tuned for specific downstream applications such as programming assistance or customized conversational AI applications.

### Out-of-Scope Use

The model should not be used for generating harmful content, spreading misinformation, or any other malicious activities.

## Bias, Risks, and Limitations

This model may inherit biases present in the training datasets, which could affect its performance on certain tasks or subpopulations.

### Recommendations

Users should be aware of the risks, biases, and limitations of the model.

## How to Get Started with the Model

Use the code below to get started with the model.

```python
from llamafactory.chat import ChatModel
from llamafactory.extras.misc import torch_gc

args = dict(
  model_name_or_path="unsloth/llama-3-8b-Instruct-bnb-4bit",
  adapter_name_or_path="llama3_lora",
  template="llama3",
  finetuning_type="lora",
  quantization_bit=4,
)
chat_model = ChatModel(args)

messages = []
print("Welcome to the CLI application, use `clear` to remove the history, use `exit` to exit the application.")
while True:
  query = input("\nUser: ")
  if query.strip() == "exit":
    break
  if query.strip() == "clear":
    messages = []
    torch_gc()
    print("History has been removed.")
    continue

  messages.append({"role": "user", "content": query})
  print("Assistant: ", end="", flush=True)

  response = ""
  for new_text in chat_model.stream_chat(messages):
    print(new_text, end="", flush=True)
    response += new_text
  print()
  messages.append({"role": "assistant", "content": response})

torch_gc()
```

## Recommended Shards

### Summary

Based on the results from testing various shards, the following model shards are recommended for generating high-quality code:

1. **`model-00004-of-00009.safetensors`**
2. **`model-00007-of-00009.safetensors`**
3. **`model-00009-of-00009.safetensors`**

These shards demonstrated the most complete and relevant code generation capabilities during our tests.

### Shard Details

<details>
<summary>Click to expand details for each shard</summary>

#### Shard: `model-00004-of-00009.safetensors`

- **Code Generation**: Successfully generated the `calculate_sum_of_squares` function with complete logic and detailed comments.
- **Use Case**: Ideal for scenarios requiring well-documented and complete code implementations. Particularly useful when detailed function descriptions and accurate logic are essential.

#### Shard: `model-00007-of-00009.safetensors`

- **Code Generation**: Generated the `calculate_sum_of_squares` function with full implementation and correct output.
- **Use Case**: Suitable for applications where precise code implementation is critical. Provides a robust solution for generating functional code snippets.

#### Shard: `model-00009-of-00009.safetensors`

- **Code Generation**: Produced a fully implemented `calculate_sum` function with clear logic and comments.
- **Use Case**: Best for tasks that require complete code snippets with proper implementation. Ensures high accuracy in generating code that adheres to the specified requirements.

</details>

---

## Usage Recommendations

### For Code Generation Tasks

- **General Code Generation**: Use any of the recommended shards for reliable and accurate code generation.
- **Documentation and Comments**: If your primary goal is to generate code with detailed comments and documentation, prefer **`model-00004-of-00009.safetensors`** and **`model-00007-of-00009.safetensors`**.

### For Specific Requirements

- **Basic Functionality**: If you only need the core functionality of code without extensive comments, **`model-00009-of-00009.safetensors`** is highly recommended.
- **Detailed Explanations**: For generating code with comprehensive explanations and detailed comments, **`model-00004-of-00009.safetensors`** and **`model-00007-of-00009.safetensors`** are preferable.

---

## Conclusion

Based on the performance observed, **`model-00004-of-00009.safetensors`**, **`model-00007-of-00009.safetensors`**, and **`model-00009-of-00009.safetensors`** are the most effective shards for generating high-quality code from the dataset. Depending on your specific needs—whether you prioritize detailed comments or basic functionality—select the shard that best aligns with your requirements.

For further customization or specific use cases, feel free to test additional shards or combinations to find the optimal model configuration for your project.

---

## Training Details

### Training Data

- **Datasets Used:** [`vicgalle/alpaca-gpt4`](https://huggingface.co/datasets/vicgalle/alpaca-gpt4) and [`sahil2801/CodeAlpaca-20k`](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k)
- **Preprocessing:** No specific preprocessing was applied to the training data.

### Training Procedure

<details>
<summary>Click to expand the Training Hyperparameters section</summary>

#### Training Hyperparameters

- **Training regime:** fp16 mixed precision
- **Batch size:** 2
- **Gradient Accumulation Steps:** 4
- **Learning Rate:** 5e-5
- **Epochs:** 3.0

Sample Training Configuration:
```python
import json

args = dict(
  stage="sft",                        
  do_train=True,
  model_name_or_path="unsloth/llama-3-8b-Instruct-bnb-4bit",
  dataset="identity,alpaca_gpt4_data,code_alpaca_20k",             
  template="llama3",                     
  finetuning_type="lora",                   
  lora_target="all",                     
  output_dir="llama3_lora",                  
  per_device_train_batch_size=2,
  per_device_eval_batch_size=2,
  max_steps=400,
  logging_steps=10,
  save_steps=100,
  save_total_limit=3,
  learning_rate=5e-5,
  max_grad_norm=0.3,
  weight_decay=0.,
  warmup_ratio=0.03,
  lr_scheduler_type="cosine",
  fp16=True,                          
  gradient_accumulation_steps=4,
)
args = json.dumps(args, indent=2)
print(args)
```
</details>

---

## Environmental Impact

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute).

- **Hardware Type:** Nvidia GeForce RTX 2060 & Nvidia Tesla T4
- **Hours used:** Approx. 50 mins
- **Cloud Provider:** Google Colab
- **Carbon Emitted:** Approximately very small amount of ~can be ignored kg~ CO2


This estimation accounts for model fine-tuning and inference runs, with a goal of providing an understanding of the environmental impact associated with the deployment and usage of the model.
## Technical Specifications

### Model Architecture and Objective

The model is based on the LLaMA-3-8B-Instruct architecture and is fine-tuned for specific tasks including text generation, code generation, and language understanding.

### Compute Infrastructure

#### Hardware

- **Type:** Nvidia GeForce RTX 2060 and Nvidia Tesla T4
- **Operating System:** Ubuntu 21, windows11 & Google Colab
- **Environment:** Google Colab Pro

#### Software

- **Frameworks:** PyTorch, Transformers

## Citation

**BibTeX:**

### 1. **LLaMA Model**:

```bibtex
@article{touvron2023llama,
  title={LLaMA: Open and Efficient Foundation Language Models},
  author={META, Touvron, Hugo and others},
  journal={arXiv preprint arXiv:2302.13971},
  year={2023},
  url={https://arxiv.org/abs/2302.13971}
}
```

### 2. **Transformers Library**:

```bibtex
@article{wolf2019transformers,
  title={Transformers: State-of-the-Art Natural Language Processing},
  author={Wolf, Thomas and others},
  journal={arXiv preprint arXiv:1910.03771},
  year={2019},
  url={https://arxiv.org/abs/1910.03771}
}
```

### 3. **Hugging Face Hub**:

```bibtex
@misc{huggingface,
  title={Hugging Face Hub},
  author={{Hugging Face}},
  year={2020},
  url={https://huggingface.co}
}
```

### 4. **Data Sets**:

#### Alpaca-GPT4:

```bibtex
@misc{vicgalle2023alpaca,
  title={Alpaca-GPT4: A dataset for training conversational models},
  author={Victor Gallego},
  year={2024},
  url={https://huggingface.co/datasets/vicgalle/alpaca-gpt4}
}
```

#### CodeAlpaca-20k:

```bibtex
@misc{sahil2023codealpaca,
  title={CodeAlpaca-20k: A dataset for code generation models},
  author={Sahil Chaudhary},
  year={2023},
  url={https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k}
}
```

### 5. **GPT-4-LLM**:

```bibtex
@misc{instruction2023gpt4,
  title={Instruction-Tuning with GPT-4},
  author={Baolin Peng*, Chunyuan Li*, Pengcheng He*, Michel Galley, Jianfeng Gao (*Equal Contribution)},
  year={2023},
  url={https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM}
}
```

## Contact and Support

For any questions, feedback, or support requests, please contact [Ian J](https://github.com/IanJ332).
