<div align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="https://upload.wikimedia.org/wikipedia/commons/9/94/USC_Trojans_logo.svg" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">DSCI566 - Deep Learning Project</h3>
</div>

---

## Table of Contents
- [About the Project](#about-the-project)
- [Built With](#built-with)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Approach](#approach)
- [Results](#results)
  - [Finance](#finance)
  - [Trivia](#trivia)
  - [Common Sense Reasoning](#common-sense-reasoning)
  - [Math](#math)
- [Setup Instructions](#setup-instructions)

---

## About the Project
This repository contains the code and results for the DSCI566 Deep Learning project. The project focuses on knowledge distillation to train compact models while maintaining high accuracy across various tasks, such as finance sentiment analysis, commonsense reasoning, and trivia question answering.

---

### Built With

* [![Python][python.js]][python-url]
* [![Hugging Face][huggingface.js]][huggingface-url]
* [![Google Cloud Platform][gcp.js]][gcp-url]

---

## Project Structure
The repository is organized as follows:

- **config/**
  - `carc_setup.sh` - Script for CARC environment setup
  - `ds_config.json` - DeepSpeed configuration
  - `environment-mac.yml` - Conda environment setup for macOS
  - `hyperparameters.yaml` - YAML file for model hyperparameters
  - `lora_config.json` - Configuration for LoRA fine-tuning

- **docs/**
  - **Literature_Review/** - Includes project reports (proposal, final report, etc.)
  - `GCP_README.md` - Instructions for setting up and using GCP resources

- **results/**
  - **training_graphs/** - PNG graphs showing training progress
  - **training_reports/** - Text files with training metrics and summaries
  - `README.md` - Description of results and key findings

- **scripts/**
  - **data_preprocessing/**
    - **tokenizers/** - Custom tokenizers used in the project
    - `data_preprocessing_flan_aqua.py` - AquaRat-specific preprocessing
  - **training/**
    - `bert_training.py` - Main script for training on the Phrasebank dataset
    - `flan_training.py` - Script for Flan-T5 training on AquaRat
    - `callback.py` - Custom callbacks for training
    - `util.py` - Utility functions used across the training scripts

- `.gitignore` - Specifies files and directories to ignore in version control
- `README.md` - Main README file with project details
- `requirements.txt` - Dependencies required for the project
- `requirements-vm.txt` - VM-specific dependencies

### Notes
- **Config directory**: Holds all configuration files for setting up environments and model parameters.
- **Docs directory**: Contains documentation and reports to understand the project's objectives and progress.
- **Scripts directory**: Focuses on data preprocessing and training scripts, categorized for clarity.
- **Results directory**: Includes all outputs from the training, such as graphs and logs.

---

## Dataset

The project leverages diverse datasets to evaluate the knowledge distillation approach across multiple domains:

- **Winogrande**: A dataset with 44,000 multiple-choice questions for pronoun disambiguation, testing common sense reasoning.
- **Social IQA**: Contains 33,000 questions evaluating logical reasoning in social contexts.
- **Trivia QA**: A dataset of 650,000 questions assessing knowledge synthesis from historical and trivia contexts.
- **Financial PhraseBank**: Comprising 4,840 sentences from financial news articles annotated for sentiment (positive, negative, or neutral).
- **AquaRat**: A dataset for assessing mathematical reasoning, challenging models with multi-step problem-solving.

---

## Approach

This project investigates the feasibility of knowledge distillation to train smaller, efficient student models from large teacher models across various domains.

### Key Components:

1. **Multi-Teacher Strategy**: In the finance domain, a multi-teacher framework was employed using FinBERT, RoBERTa, and BaseBERT. Each teacher provides complementary expertise:
   - **FinBERT**: Specializes in financial sentiment.
   - **RoBERTa**: Enhances linguistic capabilities.
   - **BaseBERT**: Contributes general knowledge.

2. **Loss Functions**:
   - **KL-Divergence Loss**: Aligns the output distributions between student and teacher models.
   - **Hidden-Layer MSE Loss**: Matches intermediate hidden states between the models, enhancing learning transfer.

This combination ensures robust performance, as demonstrated by the student model surpassing some teachers in accuracy while reducing inference time by nearly half in the finance domain.

---

## Results

The evaluation highlights the effectiveness of knowledge distillation across multiple domains:

### Finance
- The distilled DistilBERT model outperformed some teacher models, such as RoBERTa, with an accuracy of **83%** and an F1 score of **0.83**.
- Despite its smaller size, DistilBERT achieved similar accuracy to FinBERT, while reducing inference time by **50%**, showcasing a balance between efficiency and performance.


| **Role**    | **Model**                 | **Accuracy** | **F1 Score** | **Inference Time (ms/sample)** |
|-------------|---------------------------|--------------|--------------|---------------------------------|
| **Teacher** | FinBERT                   | 0.86         | 0.84         | 10.21                           |
| **Teacher** | RoBERTa Base              | 0.74         | 0.74         | 19.62                           |
| **Teacher** | BERT Base                 | 0.78         | 0.83         | 11.17                           |
| **Student** | DistilBERT (Baseline)     | 0.33         | 0.32         | 5.96                            |
| **Student** | DistilBERT (Dataset)      | 0.79         | 0.79         | 5.96                            |
| **Student** | **DistilBERT (Distilled)** | **0.83**     | **0.83**     | **5.96**                        |

> **Note:** The bolded row represents the best-performing student model, distilled using the knowledge distillation process. It achieved high accuracy and F1 score comparable to the teacher models while significantly reducing inference time.

### Trivia
- The student model, T5-Small, achieved accuracy on par with its teacher, T5-Large, demonstrating the ability to retain knowledge from larger models with fewer parameters.
- The reduced size significantly improved inference speed without compromising prediction accuracy.

| **Model Name**                | **F1 Score** | **Precision** | **Recall** | **Questions/Sec** |
|-------------------------------|--------------|---------------|------------|--------------------|
| T5 Small (Baseline)           | 0.60         | 0.48          | 0.81       | 10.79             |
| T5 Base (Teacher 1)           | 0.76         | 0.68          | 0.88       | 10.48             |
| T5 Large (Teacher 2)          | 0.84         | 0.82          | 0.85       | 5.14              |
| T5 Small (Stage 1 Student)    | **0.84**     | 0.76          | **0.93**   | **14.98**         |
| T5 Small (Stage 2 Student)    | **0.84**     | 0.76          | 0.92       | **14.96**         |

### Common Sense Reasoning
- T5-Large, as the student, maintained performance levels close to its teacher, T5-XL, highlighting the successful transfer of reasoning capabilities while maintaining a smaller model size.

| **Model Name**                | **Accuracy** | **Seconds/Question** |
|-------------------------------|--------------|-----------------------|
| T5 XL (Teacher)               | 0.81         | 12.5                 |
| T5 Large (Stage 1 Student)    | 0.72         | 5.1                  |
| T5 Large (Stage 2 Student)    | **0.79**     | **5.3**              |

### Math
- On the AquaRat dataset, the student model surpassed the teacher in validation accuracy (27.17% vs. 23.23%).
- However, the results revealed a limitation for tasks requiring long chains of reasoning, as smaller models lack the parameters to effectively process extensive thought processes.

These results emphasize the potential of knowledge distillation to train compact, high-performing models across diverse tasks, though some domains with higher complexity, such as math, may require further exploration.

---

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd ScalingDown
   ```
   
2. Install required dependencies:
   ```bash
   conda env create -f config/environment-mac.yml
   conda activate scaling-down
   ```
   
3. Configure the project:\
   - Modify config/hyperparameters.yaml and ds_config.json as needed for your experiment.


4. Run the training scripts:
   - For DistilBERT Training (e.g., Finance Domain):
    ```bash
    python bert_training.py --dataset phrasebank --data_portion 1 --output_report ../output_report
    ```
    - Flan-T5 Training (e.g., Math Domain)::
    ```bash
    python flan_training.py --dataset aquarat --data_portion 1 --output_report ../output_report
    ```

[python.js]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[python-url]: https://www.python.org/
[huggingface.js]: https://img.shields.io/badge/Hugging%20Face-FF7A00?style=for-the-badge&logo=huggingface&logoColor=white
[huggingface-url]: https://huggingface.co/
[gcp.js]: https://img.shields.io/badge/Google%20Cloud%20Platform-4285F4?style=for-the-badge&logo=googlecloud&logoColor=white
[gcp-url]: https://cloud.google.com/
   