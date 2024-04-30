# Enhanced Text-to-Image Synthesis via Prompt Optimization

## Introduction
This repository contains the source code and documentation for our project on systematic prompt optimization for enhanced text-to-image synthesis, developed at University College London. Our work introduces a novel approach using in-context gradient descent within a Multi-Agent framework to improve the quality of images generated from textual prompts.

## Project Overview
Our project leverages a sophisticated Multi-Agent system to dynamically evolve the design of prompts used in text-to-image synthesis models. By integrating a professional prompts database(Lexicas) and utilizing advanced scoring functions like the Human Preference Score v2 (HPSv2), our system refines prompts to produce images that closely align with professional standards and human aesthetic preferences.

### Key Features
- **Prompt Optimization Algorithm**: Utilizes in-context gradient descent to refine prompts systematically.
- **Multi-Agent Framework**: Employs multiple agents to iteratively improve the quality of prompts based on performance feedback.
- **Professional Prompts Database**: Guides the optimization process with high-caliber prompts to ensure the generation of professional-grade images.
- **Advanced Scoring Metrics**: Leverages HPSv2 to evaluate image quality, facilitating continuous improvement in prompt design.

## Technologies Used
- GPT-3.5 Turbo via OpenAI Assistant API
- Human Preference Score v2 (HPSv2)
- Python for scripting and automation




# To Run in Colab Notebooks: 

```shell
!git clone https://github.com/slark-prime/MAAI_Optimizer.git
%cd MAAI_Optimizer
!git clone https://github.com/tgxs002/HPSv2.git
%cd HPSv2
!pip install -e .
%cd ..
!pip install hpsv2
!pip install openai
!pip install diffusers

```
```
!python run.py
```
# Experiments
- [x] UCB batch 1 with lexica 
- [x] UCB batch 3 with lexica 
- [x] UCB batch 5 with lexica 
- [x] UCB batch 3 without lexica 
- [x] greedy batch 3 with lexica
- [x] epsilon greedy batch 3 with lexica
- [x] gpt3.5 baseline

## Project Paper
For detailed information about the methodologies and technologies used in this project, refer to our [Batch-Instructed Gradient for Prompt Evolution: Systematic Prompt Optimization for Enhanced Text-to-Image Synthesis](Text-to-Image%20Synthesis%20via%20Prompt%20Optimization.pdf).
