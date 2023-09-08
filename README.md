[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)


# GPT-3: Few-Shot Learning for Language Models
The Base LLM behind GPT-3 and the paper 

## 💻 Installation

`pip install gpt3-torch`

---


## Code Example

Here's an illustrative code snippet that showcases GPT-3 in action:


```python
import torch
from gpt3.gp3 import GPT3

# Generate a random input sequence
x = torch.randint(0, 256, (1, 1024)).cuda()

# Initialize GPT-3 model
model = GPT3()

# Pass the input sequence through the model
output = model(x)
```


### 📚 Training

```python
from gpt3 import train

train()

```

For further instructions, refer to the [Training SOP](DOCs/TRAINING.md).


1. Set the environment variables:
   - `ENTITY_NAME`: Your wandb project name
   - `OUTPUT_DIR`: Directory to save the weights (e.g., `./weights`)
   - `MASTER_ADDR`: For distributed training
   - `MASTER_PORT` For master port distributed training
   - `RANK`- Number of nodes services
   - `WORLD_SIZE` Number of gpus

2. Configure the training:
   - Accelerate Config
   - Enable Deepspeed 3
   - Accelerate launch train_distributed_accelerate.py

For more information, refer to the [Training SOP](DOCs/TRAINING.md).




---

Welcome to the repository for GPT-3: Few-Shot Learning for Language Models! This repository provides code examples and insights related to the groundbreaking paper "Language Models are Few-Shot Learners" by Tom B. Brown et al. Explore the potential of GPT-3, a language model with 175 billion parameters, and its remarkable few-shot learning capabilities. Below, we provide an overview of key concepts, practical code snippets, and the paper's findings.

## Introduction

In recent years, Natural Language Processing (NLP) has witnessed remarkable progress through pre-training language models on vast text corpora and fine-tuning them for specific tasks. However, these models still demand substantial task-specific data to excel. This paper introduces a paradigm shift by unveiling the concept of few-shot learning for language models. Discover how the scale of the model impacts its performance, akin to humans learning from just a few examples or simple instructions.

## Methodology

This paper introduces GPT-3, an autoregressive language model with a groundbreaking scale of 175 billion parameters. The authors assess GPT-3's few-shot learning capabilities by subjecting it to various tasks without any gradient updates or fine-tuning. The model's understanding of tasks and demonstrations is achieved solely through text interactions.

## Results

The paper presents compelling results highlighting GPT-3's prowess in few-shot learning:

- **Translation**
- **Question-answering**
- **Cloze tasks**
- **On-the-fly reasoning**
- **Domain adaptation tasks**

Furthermore, GPT-3 excels in tasks that involve unscrambling words, incorporating novel words into sentences, and performing 3-digit arithmetic. While demonstrating its potential, the paper acknowledges areas where GPT-3's few-shot learning encounters challenges, opening avenues for future enhancement. Additionally, methodological concerns related to training language models on extensive web corpora are discussed.


# Dataset Strategy
Here is a table of the datasets used in the paper with metadata and source links:

| Dataset | Task Type | # Examples | Metric | Source Link |
|-|-|-|-|-| 
| Penn Tree Bank (PTB) | Language Modeling | 42k | Perplexity | https://catalog.ldc.upenn.edu/LDC99T42 |
| LAMBADA | Language Modeling | 10k | Accuracy, Perplexity | https://zenodo.org/record/2630551 |  
| HellaSwag | Sentence Completion | 100k | Accuracy | https://rowanzellers.com/hellaswag/ |
| StoryCloze | Sentence Completion | 20k | Accuracy | https://www.cs.rochester.edu/nlp/rocstories/ |
| Natural Questions | QA | 104k | F1 | https://ai.google.com/research/NaturalQuestions |  
| TriviaQA | QA | 95k | Accuracy | http://nlp.cs.washington.edu/triviaqa/ |
| WebQuestions | QA | 5.9k | Accuracy | https://www.microsoft.com/en-us/download/details.aspx?id=52763 |
| X→En WMT | Translation | 3k | BLEU | http://www.statmt.org/wmt14/translation-task.html |
| Winograd Schema Challenge | Commonsense QA | 273 | Accuracy | https://cs.nyu.edu/faculty/davise/papers/WinogradSchemas/WS.html |
| PIQA | Commonsense QA | 1.8k | Accuracy | https://allenai.org/data/physical-iq |  
| ARC | Commonsense QA | 14k | Accuracy | http://data.allenai.org/arc/ |
| OpenBookQA | Commonsense QA | 5k | Accuracy | https://allenai.org/data/open-book-qa |
| QuAC | Reading Comprehension | 98k | F1 | https://quac.ai/ |
| RACE | Reading Comprehension | 28k | Accuracy | http://www.qizhexie.com/data/RACE_leaderboard.html |
| SQuAD v2 | Reading Comprehension | 130k | F1 | https://rajpurkar.github.io/SQuAD-explorer/ |  
| CoQA | Reading Comprehension | 127k | F1 | https://stanfordnlp.github.io/coqa/ |
| DROP | Reading Comprehension | 77k | F1 | https://allennlp.org/drop |  
| SuperGLUE | GLUE Benchmark | 32k | Average Score | https://super.gluebenchmark.com/ |
| Winogrande | Commonsense QA | 1.2k | Accuracy | https://pilehvar.github.io/winogrande/ |
| Arithmetic | Synthetic | 20k | Accuracy | https://github.com/openai/gpt-3 |
| Word Scrambling | Synthetic | 100k | Accuracy | https://github.com/openai/gpt-3 |
| SAT Analogies | Synthetic | 374 | Accuracy | https://github.com/openai/gpt-3 |


## Conclusion

The study concludes that scaling up model size, as exemplified by GPT-3, substantially elevates few-shot learning capabilities. GPT-3 achieves competitive results compared to state-of-the-art fine-tuning approaches. The authors delve into the broader implications of GPT-3's capabilities, including its potential to generate human-like text. The paper emphasizes the need for ongoing research to address challenges in challenging few-shot learning tasks and to address methodological concerns associated with large web corpora training.

For a comprehensive understanding of the paper's methodologies, insights, and findings, refer to the original publication: [Language Models are Few-Shot Learners](https://doi.org/arXiv.2005.14165).

If you find this repository valuable, consider starring it or contributing to foster continual exploration and discourse in the field of NLP and few-shot learning.


# Citation
```latex
@misc{2005.14165,
Author = {Tom B. Brown and Benjamin Mann and Nick Ryder and Melanie Subbiah and Jared Kaplan and Prafulla Dhariwal and Arvind Neelakantan and Pranav Shyam and Girish Sastry and Amanda Askell and Sandhini Agarwal and Ariel Herbert-Voss and Gretchen Krueger and Tom Henighan and Rewon Child and Aditya Ramesh and Daniel M. Ziegler and Jeffrey Wu and Clemens Winter and Christopher Hesse and Mark Chen and Eric Sigler and Mateusz Litwin and Scott Gray and Benjamin Chess and Jack Clark and Christopher Berner and Sam McCandlish and Alec Radford and Ilya Sutskever and Dario Amodei},
Title = {Language Models are Few-Shot Learners},
Year = {2020},
Eprint = {arXiv:2005.14165},
}
```