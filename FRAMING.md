# Framing: Rawaj Pay mule detection

## Decision

Rawaj Pay processes around 3,000 instant transfers per day, and five fraud analysts are responsible for reviewing suspicious cases. For each transfer, the decision is whether it goes to a human analyst for review or is allowed through. The system does not block anyone automatically. It ranks transfers by risk so analysts can prioritize the cases most likely to involve mule activity. False positives waste limited analyst time, while false negatives allow mule activity to go undetected.

## Metric

Since analyst capacity is limited, precision at k is useful for measuring how many of the highest-ranked cases are actually positive, where k is the number of transfers analysts can review daily. I would also track precision and recall at the chosen threshold and use expected cost, calculated as `FP × cost_fp + FN × cost_fn`, to select the threshold. The real cost values have to come from Dina's team. Accuracy alone is unsuitable because mule cases are rare, and a system that flags nothing could still achieve high accuracy.

## Baseline

Before using a model, Rawaj Pay could prioritize transfers using simple rules, such as unusually high amounts or transfers to new recipients above a specified amount. A model should outperform this baseline in precision at k and expected cost, using realistic cost estimates and the available analyst capacity. If it does not provide a measurable improvement, the additional complexity is not justified.

## Classical ML or LLM

I would start with classical machine learning, such as logistic regression or gradient-boosted trees, because transaction data is structured and the task requires fast, reproducible risk scores. Raw transaction data should remain within the company's approved environment rather than being sent to an external hosted LLM. An LLM could help summarize flagged cases for analysts, but it should not replace the risk-scoring model. Before training, I would audit the extract for data-quality issues and target leakage, including fields such as `case_closed_at`, which may be recorded after an investigation.
