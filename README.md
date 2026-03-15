# Behavioral Bias in Data Science

## Overview

Human decision-making is not purely rational. When analysts interact with data, they often introduce **behavioral biases** that distort interpretation, model selection, and conclusions.

This project investigates how **behavioral biases influence data analysis and machine learning workflows**. Through simulations and experiments, we demonstrate how biases such as confirmation bias, survivorship bias, and overconfidence bias can affect model evaluation and decision-making.

The goal of this project is to:

* illustrate how behavioral biases emerge in data-driven environments
* simulate biased decision processes
* evaluate how bias impacts model performance and interpretation

---

## Objectives

This project focuses on three main objectives:

1. **Simulate behavioral biases**

   * Create controlled environments where biased decisions occur.

2. **Analyze bias effects**

   * Measure how biased model selection affects performance.

3. **Demonstrate real-world implications**

   * Show how bias can lead to misleading conclusions in data science.

---

## Types of Bias Studied

The project focuses on several common cognitive biases:

### Confirmation Bias

Analysts favor models or results that confirm their initial hypotheses.

### Survivorship Bias

Only successful outcomes are observed while failures are ignored.

### Overconfidence Bias

Decision-makers overestimate their predictive abilities.

---

## Project Structure

```
behavioral-bias
│
├── data
│   ├── raw
│   ├── processed
│   └── external
│
├── notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_bias_simulation.ipynb
│   ├── 03_model_selection_bias.ipynb
│   └── 04_results_visualization.ipynb
│
├── src
│   ├── data
│   ├── simulation
│   ├── models
│   ├── metrics
│   └── visualization
│
├── experiments
├── README.md
└── requirements.txt
```

---

## Methodology

The project uses two main approaches:

### Simulation Experiments

Synthetic datasets are generated to create controlled scenarios where behavioral bias appears during analysis.

Examples include:

* biased model selection
* cherry-picked evaluation metrics
* ignoring negative results

### Data Analysis

Experimental results are analyzed using statistical metrics and visualization techniques to understand the magnitude of bias effects.

---

## Experiments

The following experiments are implemented:

### Experiment 1: Confirmation Bias in Model Selection

Simulate analysts choosing models that support their expectations rather than the best-performing model.

### Experiment 2: Survivorship Bias in Strategy Evaluation

Demonstrate how ignoring failed strategies leads to overly optimistic conclusions.

### Experiment 3: Overconfidence in Predictive Models

Analyze how excessive confidence in models leads to poor generalization.

---

## Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Jupyter Notebook

---

## How to Run

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Launch notebooks:

```
jupyter notebook
```

3. Run experiments:

```
python experiments/experiment_01_confirmation_bias.py
```

---

## Expected Outcomes

This project aims to demonstrate that:

* behavioral biases can significantly distort data-driven decisions
* model evaluation can be manipulated unintentionally
* rigorous methodology is necessary to avoid biased conclusions

