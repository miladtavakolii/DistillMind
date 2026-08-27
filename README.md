# DistillMind 🧠

**DistillMind** is an experimental research project for building a compact Persian language model specialized in **mental-state and emotion-related text classification** through a multi-stage training and knowledge distillation pipeline.

The project explores how a general-purpose language model can be adapted to understand Persian emotional and mental-state signals, while transferring task-specific knowledge into a smaller and more efficient model.

---

## ✨ Overview

Large Language Models (LLMs) can perform a wide range of NLP tasks, but deploying them for a specialized task can be unnecessarily expensive in terms of memory and inference cost.

DistillMind investigates a different approach:

> **Teach a capable language model the task first, then distill the learned knowledge into a smaller specialized model.**

The current pipeline focuses on Persian text and uses a Gemma-based model with parameter-efficient fine-tuning and 4-bit quantization.

The project consists of several stages:

```text
Raw / Generated Data
        │
        ▼
┌──────────────────────┐
│ Dataset Construction │
└──────────┬───────────┘
           │
           ▼
┌────────────────────────────┐
│ Emotion Representation     │
│ Pretraining                │
└──────────┬─────────────────┘
           │
           ▼
┌────────────────────────────┐
│ Task-Specific Distillation │
│ & Fine-Tuning              │
└──────────┬─────────────────┘
           │
           ▼
┌──────────────────────┐
│ Specialized Gemma   │
│ Model                │
└──────────┬───────────┘
           │
           ▼
       Inference
```

---

## 🎯 Task

The current task focuses on classifying Persian text according to mental-state-related categories.

The inference interface currently uses three labels:

* `fear`
* `withdrawal`
* `other`

The model is prompted to provide both:

1. A short explanation of why the text belongs to a category.
2. The final classification label.

The expected output format is JSON:

```json
{
  "reason": "Brief explanation of the classification.",
  "label": "fear"
}
```

---

## 🧪 Training Pipeline

### 1. Dataset Generation

DistillMind contains utilities for constructing training data using LLM-based generation.

The dataset pipeline is located under:

```text
dataset/
├── dataset_creator.py
├── llm_data_generator.py
├── run_generation.py
└── prompts/
    └── phase2/
```

The generation pipeline can be used to create task-specific examples containing:

```text
text
reason
label
```

These examples are subsequently transformed into conversational training samples.

---

### 2. Emotion Representation Pretraining

The project includes an **Emotion Representation Pretraining** stage.

This stage aims to adapt the model's internal representations toward emotional and mental-state-related information before performing the final task-specific training.

The corresponding experiment is available in:

```text
Emotion Representation Pretraining.ipynb
```

The notebook uses the Unsloth training stack and supports efficient fine-tuning of Gemma-based models.

---

### 3. Task-Specific Distillation

After the representation-pretraining stage, the project performs task-specific training using a dedicated dataset.

The training data is converted into a conversational format containing:

```text
System
   ↓
Task instructions

User
   ↓
Persian text

Assistant
   ↓
Reason + classification label
```

For example:

```json
{
  "conversations": [
    {
      "role": "system",
      "content": "You are a Persian mental-state classifier..."
    },
    {
      "role": "user",
      "content": "متن فارسی ..."
    },
    {
      "role": "assistant",
      "content": "{\"reason\":\"...\",\"label\":\"other\"}"
    }
  ]
}
```

The task-specific training experiment is available in:

```text
Task-specific Distillation Fine-tuning.ipynb
```

The notebook uses the Gemma 3 chat template and prepares the dataset for instruction-style fine-tuning.

---

## ⚡ Efficient Fine-Tuning

DistillMind uses several techniques to reduce the computational requirements of training:

* **LoRA / PEFT**
* **4-bit quantization**
* **Unsloth**
* **PyTorch**
* **Hugging Face Transformers**
* **bitsandbytes**

The training notebook demonstrates loading the model using 4-bit quantization and Unsloth's optimized model interface.

This makes experimentation with relatively large language models possible on more limited GPU hardware.

---

## 📂 Project Structure

```text
DistillMind/
│
├── data/
│   └── ...
│
├── dataset/
│   ├── prompts/
│   │   └── phase2/
│   ├── dataset_creator.py
│   ├── llm_data_generator.py
│   └── run_generation.py
│
├── llm_engine/
│   ├── base.py
│   ├── engine.py
│   ├── gemini_client.py
│   └── ollama_client.py
│
├── utils/
│   └── ...
│
├── Emotion Representation Pretraining.ipynb
├── Task-specific Distillation Fine-tuning.ipynb
├── inference.py
│
├── pyproject.toml
└── uv.lock
```

The repository currently separates dataset generation, LLM interaction, model training, utilities, and inference into dedicated components.

---

## 🛠️ Installation

### Requirements

* Python `3.12+`
* NVIDIA GPU recommended for training
* CUDA-compatible PyTorch installation
* Hugging Face account/token when required by the selected model
* Optional API credentials for LLM-based dataset generation

The project currently declares Python `>=3.12` and depends on packages including PyTorch, Transformers, PEFT, Unsloth, Accelerate, bitsandbytes, Datasets, and Google GenAI.

### Using `uv`

Clone the repository:

```bash
git clone https://github.com/miladtavakolii/DistillMind.git
cd DistillMind
```

Install the dependencies:

```bash
uv sync
```

Activate the environment:

```bash
source .venv/bin/activate
```

---

## 🗂️ Dataset Generation

The dataset generation utilities are located in:

```text
dataset/
```

Before generating data, configure the required LLM provider and prompts.

The project currently contains LLM clients for:

* Google Gemini
* Ollama

implemented under:

```text
llm_engine/
```

The generation pipeline can then be started through:

```bash
python dataset/run_generation.py
```

> Dataset generation may require an API key depending on the selected LLM backend.

---

## 🧑‍🏫 Training

The main training experiments are provided as Jupyter notebooks.

Start Jupyter:

```bash
jupyter notebook
```

Then open:

```text
Emotion Representation Pretraining.ipynb
```

and subsequently:

```text
Task-specific Distillation Fine-tuning.ipynb
```

The current task-specific experiment uses a JSONL dataset and creates a train/validation split before converting examples into the Gemma conversational format.

---

## 🔍 Inference

After training and saving the model, inference can be performed using:

```bash
python inference.py
```

The inference wrapper automatically selects CUDA when available and otherwise falls back to CPU.

By default, it expects the model to be available at:

```text
./model
```

The inference implementation uses Hugging Face `AutoTokenizer` and `AutoModelForCausalLM`, with configurable generation parameters such as:

* `max_new_tokens`
* `temperature`
* `top_p`
* `top_k`

and supports streaming generation.

Example:

```text
Chat ready. Type "exit" to quit.

>> از وقتی این اتفاق افتاده خیلی می‌ترسم و ترجیح می‌دم از همه دور باشم.
```

The model can then return a structured classification containing the predicted label and its reasoning.

---

## 🤝 Contributing

Contributions, experiments, and research ideas are welcome.

If you find a bug or have an idea for improving the training pipeline, feel free to open an issue or submit a pull request.

---

## 📄 License

License information will be added to the repository.

---

## 👤 Author

**Milad Tavakoli**

GitHub: [@miladtavakolii](https://github.com/miladtavakolii)

---

## ⭐ Acknowledgements

This project builds on the open-source ecosystem around:

* [Hugging Face Transformers](https://github.com/huggingface/transformers)
* [Hugging Face Datasets](https://github.com/huggingface/datasets)
* [PEFT](https://github.com/huggingface/peft)
* [Unsloth](https://github.com/unslothai/unsloth)
* [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes)
* [PyTorch](https://github.com/pytorch/pytorch)
