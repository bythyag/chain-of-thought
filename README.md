## Chain of Thought (CoT) Experiments

This repository contains implementation and analysis of Chain of Thought (CoT) prompting experiments across various language models and reasoning tasks. The project evaluates how different prompting strategies affect model performance on arithmetic, commonsense, and symbolic reasoning tasks.

### Overview

This codebase implements experiments to test and validate claims about Chain of Thought prompting across different model scales and reasoning tasks. It includes:

- Implementation of CoT and standard prompting evaluations
- Ablation studies on prompt components
- Analysis of model performance across different reasoning tasks
- Out-of-distribution (OOD) testing
- Comparative analysis across different model sizes

### Key Features

- Support for multiple reasoning tasks:
  - Arithmetic (GSM8K, SVAMP, ASDiv, AQuA, MAWPS)
  - Commonsense (CSQA, StrategyQA, Date, Sports, SayCan)
  - Symbolic (Last letter concatenation, Coin flip)
- Comprehensive evaluation framework
- Ablation study implementations

### Project Structure

- `src/`: Core implementation files
  - `evals.py`: Evaluation benchmarking code
  - `ablation.py`: Ablation study implementation
  - `ood.py`: Out-of-distribution testing
- `prompts/`: Prompt templates and examples
- `logs/`: Experimental results and analysis
- `sample results/`: Raw output from various model runs

### Key Findings on GPT4.1 family.

- High accuracy with GPT-4.1: Out of 200 GSM8K samples, only 4 mistakes occurred—2 due to semantic misinterpretation and 1 due to dataset error, showing strong performance.
- System prompt sensitivity: Adding strict instructions sometimes reduced generalization and led to incorrect answers, even when CoT examples alone worked.
- Model size effects: GPT-4.1-mini and nano avoided arithmetic errors but failed in sequential reasoning, assumptions, and event tracking compared to GPT-4.1.
- Ablation results: Equation-only prompting gave 93% accuracy, while reasoning-after-answer dropped to 61%, confirming that stepwise CoT reasoning is critical. In variable computer where we relaced the CoT steps with dots, we got a performance of 63%. 
- OOD robustness: On symbolic reasoning tasks (last-name concatenation, coin flip), GPT-4.1 achieved near 100% accuracy with CoT, outperforming standard prompting (98%).

### References
You can read more about reasoning in LLMs in orginal [Chain-of-Thought paper](https://arxiv.org/abs/2201.11903).
