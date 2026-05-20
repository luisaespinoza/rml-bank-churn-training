# Bank Customer Churn Risk Scorer

A lightweight, browser-deployable machine learning project for predicting bank customer churn from structured account and customer data.

This project is part of a broader **minimal deployable ML** portfolio: small, practical machine learning systems that show AI is broader than chatbots, large language models, or cloud-hosted GenAI APIs. The goal is to demonstrate a deliberate deployment tradeoff: accept a small performance loss versus a strong tabular benchmark in exchange for a smaller, simpler model that can run locally in the browser with no backend, no API keys, and no paid inference service.

## Project Summary

The model estimates the probability that a bank customer will churn, using structured features such as credit score, geography, gender, age, tenure, balance, number of products, credit-card status, active membership, and estimated salary.

The intended deployment target is a static GitHub Pages app that performs client-side inference. A user enters customer/account attributes, the app preprocesses those inputs, runs a compact neural network locally, and returns a churn-risk score.

This is not a chatbot and not a generative AI application. It is a compact predictive model for business decision support.

## Why This Project Exists

Many small businesses and non-technical stakeholders hear "AI" and assume it means large cloud systems, GPUs, chatbots, or expensive API calls. This project demonstrates a different category of AI:

- train a model offline in Python;
- export a small model artifact;
- run inference directly in the browser;
- avoid servers, API keys, and cloud inference costs;
- evaluate performance using business-relevant targeting metrics.

The main thesis is not that a small ANN must beat every tabular model. The thesis is that a minimally sized ANN can reach a useful performance band while being much easier to deploy as a static, local-inference web application. In this project, minimality is treated as an engineering objective: a small loss in benchmark performance can be worthwhile when it materially improves deployability, portability, and operational simplicity.

## Repository Role

This repository is the **training repository**. It contains the Python code used to:

- load and clean the churn dataset;
- split the data into train/validation/test sets;
- preprocess numeric and categorical features;
- train several compact ANN architectures;
- compare ANN models against non-ANN baselines;
- save model checkpoints and deployment metadata;
- export browser-oriented deployment artifacts.

A separate deployment/web repository can consume the exported model and preprocessing schema to run inference in a static GitHub Pages app.

## Dataset

Current dataset target: **Bank Customer Churn** dataset, such as the Maven Analytics / Kaggle variant.

Expected target column:

- `Exited`

Expected feature columns include:

- `CreditScore`
- `Geography`
- `Gender`
- `Age`
- `Tenure`
- `Balance`
- `NumOfProducts`
- `HasCrCard`
- `IsActiveMember`
- `EstimatedSalary`

Identifier columns are dropped if present:

- `RowNumber`
- `CustomerId`
- `Surname`

Dataset attribution:

- Provider / access point: Maven Analytics Data Playground
- Source page: https://mavenanalytics.io/data-playground/bank-customer-churn
- Direct download: https://maven-datasets.s3.amazonaws.com/Bank+Customer+Churn/Bank+Customer+Churn.zip
- Original source listed by Maven: Kaggle
- License listed by Maven: Public Domain
- Access date: 2026-05-19

See `DATASET_ATTRIBUTION.md` for dataset handling, citation, and redistribution notes.

## Modeling Approach

The project compares four compact artificial neural networks against standard tabular baselines.

### ANN model family

The ANN models are implemented in PyTorch:

| Model | Shape |
|---|---|
| Tiny | input -> 32 -> 16 -> 1 |
| Small | input -> 64 -> 32 -> 1 |
| Medium | input -> 128 -> 64 -> 32 -> 1 |
| Large | input -> 256 -> 128 -> 64 -> 32 -> 1 |

All ANN models use ReLU hidden activations and produce a single binary-classification logit.

### Non-ANN baselines

The comparison baselines are:

| Model | Purpose |
|---|---|
| Logistic Regression | Simple interpretable linear baseline |
| Random Forest | Nonlinear tree-ensemble baseline |
| XGBoost | Strong tabular performance benchmark |

The ANN does not need to beat XGBoost to be useful. XGBoost is used as a strong raw-performance ceiling, while the ANN is evaluated as a deployment-oriented model. The comparison is intentionally framed as a tradeoff: how much performance is lost when choosing a smaller model that is easier to ship as a static, local-inference browser app?

## Preprocessing

The data split is stratified by the target column.

Current split:

- 70% train
- 15% validation
- 15% test

Preprocessing rules:

- numeric columns: `StandardScaler`
- categorical columns: `OneHotEncoder(handle_unknown="ignore")`
- preprocessing is fit only on the training set;
- validation and test sets are transformed with the fitted preprocessing pipeline.

The browser deployment schema stores the numeric means/scales and categorical one-hot category order so the web app can reproduce the Python preprocessing logic.

## Training Setup

ANN training uses:

- framework: PyTorch
- loss: `BCEWithLogitsLoss`
- optimizer: Adam
- learning rate: `1e-3`
- prediction transform: `sigmoid(logit)`
- checkpoint selection: best validation loss

Current canonical ANN run:

- seed: `314`
- epochs: `15,000`

Saved ANN checkpoints are reused for comparison. They do not need to be retrained just to compare against baseline metrics.

## Metrics

The project reports both standard ML metrics and business-targeting metrics:

| Metric | Meaning |
|---|---|
| ROC-AUC | Overall ranking quality across thresholds |
| PR-AUC / Average Precision | Precision-recall quality for imbalanced churn prediction |
| Recall@Top 20% | Of all actual churners, how many are captured if the business contacts only the riskiest 20% of customers? |
| Precision@Top 20% | Among the riskiest 20% of customers, what fraction actually churned? |
| Brier Score | Probability calibration error; lower is better |
| Log Loss | Probabilistic classification loss; lower is better |
| Checkpoint Size | Practical deployment size of the saved model artifact |

For this business use case, `Recall@Top 20%` and `Precision@Top 20%` are especially important because they map directly onto a retention campaign: if the bank can contact only a limited number of customers, how well does the model prioritize the highest-risk accounts?

## Current Results

The current canonical results use the seed-314 ANN checkpoints and the seed-314 Tiny ANN vs XGBoost runtime/classification benchmark. 

### ANN checkpoint results

| Model | Type | ROC-AUC | PR-AUC | Recall@Top 20% | Precision@Top 20% | Brier Score | Log Loss | Parameters | Checkpoint Size MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Small | ANN | 0.868120 | 0.714749 | 0.619672 | 0.630000 | 0.099446 | 0.328174 | 3009 | 0.347279 |
| Tiny | ANN | 0.866515 | 0.718818 | 0.622951 | 0.633333 | 0.099371 | 0.328620 | 993 | 0.339579 |
| Large | ANN | 0.865264 | 0.708955 | 0.622951 | 0.633333 | 0.101206 | 0.332204 | 46849 | 0.515408 |
| Medium | ANN | 0.862432 | 0.705063 | 0.613115 | 0.623333 | 0.101422 | 0.334038 | 12161 | 0.382618 |

### Tiny ANN vs XGBoost, seed 314

| Model | ROC-AUC | PR-AUC | Recall@Top 20% | Precision@Top 20% | Brier Score | Log Loss |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost | 0.880337 | 0.743483 | 0.655738 | 0.666667 | 0.095025 | 0.316171 |
| Tiny ANN | 0.866515 | 0.718818 | 0.622951 | 0.633333 | 0.099371 | 0.328620 |

## Result Interpretation

XGBoost is the strongest raw tabular benchmark in the current seed-314 comparison:

- ROC-AUC: `0.880337`
- Recall@Top 20%: `0.655738`
- Precision@Top 20%: `0.666667`

The best ANN by ROC-AUC is **Small**:

- ROC-AUC: `0.868120`

The best ANN by top-20% targeting is a tie between **Tiny** and **Large**:

- Recall@Top 20%: `0.622951`
- Precision@Top 20%: `0.633333`

The **Tiny ANN** is the most deployment-aligned candidate:

- ROC-AUC: `0.866515`
- Recall@Top 20%: `0.622951`
- Precision@Top 20%: `0.633333`
- checkpoint size: `0.339579 MB`

Compared with XGBoost, Tiny gives up about `0.013823` ROC-AUC and about `3.3` percentage points of Recall@Top 20%. Precision@Top 20% is lower by about `3.3` percentage points. In exchange, Tiny provides the smallest ANN checkpoint and substantially faster model-only inference, which better matches the project goal of lightweight browser deployment.

Tiny is also only about `0.001605` ROC-AUC behind Small while tying Large for the best ANN top-20% targeting metrics and using the smallest ANN checkpoint.

## Runtime and Deployment Tradeoff

The runtime benchmark compares Tiny ANN against XGBoost using seed `314`.

This is a deliberate deployment tradeoff. XGBoost produces stronger classification metrics, while Tiny ANN provides a smaller and simpler deployment target with substantially cheaper model-only inference.

Model-only inference latency favored the Tiny ANN by a wide margin:

| Batch Size | Tiny ANN Median | XGBoost Median | Tiny Speedup |
|---:|---:|---:|---:|
| 1 | 0.062 ms | 0.320 ms | 5.2x |
| 32 | 0.067 ms | 0.468 ms | 7.0x |
| 300 | 0.087 ms | 1.484 ms | 17.2x |
| 1500 | 0.153 ms | 6.042 ms | 39.5x |

End-to-end Python inference includes preprocessing overhead, so the single-customer result is closer:

| Batch Size | Tiny ANN End-to-End | XGBoost End-to-End | Faster |
|---:|---:|---:|---|
| 1 | 3.100 ms | 2.617 ms | XGBoost |
| 32 | 3.201 ms | 2.860 ms | XGBoost |
| 300 | 3.375 ms | 3.968 ms | Tiny ANN |
| 1500 | 4.045 ms | 9.237 ms | Tiny ANN |

These are local Python CPU timings, not browser timings. They should be interpreted as a local benchmark showing the cost structure: preprocessing dominates single-record end-to-end latency, while the Tiny ANN itself is substantially cheaper to execute than XGBoost. Browser runtime should be measured separately in the deployed web app.

## Deployment Decision

Current deployment candidate: **Tiny ANN**.

Rationale:

- It is within roughly `0.014` ROC-AUC of XGBoost in the seed-314 benchmark.
- It is within about `3.3` percentage points of XGBoost on Recall@Top 20% and `3.3` percentage points on Precision@Top 20%.
- It ties the best ANN top-20% targeting performance in the seed-314 checkpoint comparison.
- It has the smallest ANN checkpoint size at roughly `0.34 MB`.
- It is substantially faster than XGBoost for model-only inference, with speedups ranging from about `5x` for single-customer inference to nearly `40x` for large batches in the local CPU benchmark.
- It best represents the project's central deployment tradeoff: a small performance hit in exchange for a minimally sized, browser-friendly model that can run without backend inference, API keys, or paid cloud services.

This choice is not based on Tiny being the strongest raw predictive model. XGBoost remains the best benchmark for tabular predictive performance. The deployment choice favors Tiny because the goal of this project is not maximum offline accuracy; it is lightweight, static, local inference with acceptable predictive performance.

If future seed sweeps show Tiny is unstable, **Small ANN** is the safer fallback because it currently has the best ANN ROC-AUC while remaining nearly the same checkpoint size.

## Artifacts

Training produces the following artifacts:

```text
artifacts/
  models/
    tiny.pt
    small.pt
    medium.pt
    large.pt
    best_model.pt

  deployment/
    best_model.onnx
    preprocessing_schema.json
    deployment_manifest.json

report/
  output_results.json
  saved_model_comparison_results.json
  saved_model_comparison_results.csv
  runtime_tiny_vs_xgboost.json
  runtime_tiny_vs_xgboost.csv
  runtime_tiny_vs_xgboost_classification.csv
```

Notes:

- `.pt` checkpoints store model weights and metrics.
- `best_model.pt` stores the selected ANN checkpoint.
- `preprocessing_schema.json` stores enough metadata for browser-side feature construction.
- `deployment_manifest.json` records selected-model metadata.
- ONNX export requires the `onnx` package. If ONNX export fails, the `.pt` checkpoints and manifest still remain useful.

## Running the Experiments

Train ANN models and save deployment artifacts:

```bash
python curr_exp.py --progress --epochs 15000 --progress-every 1000
```

Train a subset of ANN models:

```bash
python curr_exp.py --progress --epochs 15000 --models tiny small
```

Run non-ANN baselines:

```bash
python prev_exp.py
```

Compare saved ANN checkpoint metrics against saved non-ANN baseline metrics:

```bash
python compare_saved_ann_vs_non_ann.py
```

Benchmark Tiny ANN against XGBoost runtime and classification performance:

```bash
python benchmark_tiny_vs_xgboost_runtime.py --seed 314
```

The comparison scripts should load already-saved ANN checkpoints from `artifacts/models/*.pt` and should not retrain the ANN models.

## Suggested Project Structure

```text
.
├── data/
│   └── Bank_Churn.csv
├── artifacts/
│   ├── models/
│   │   ├── tiny.pt
│   │   ├── small.pt
│   │   ├── medium.pt
│   │   ├── large.pt
│   │   └── best_model.pt
│   ├── baselines/
│   │   └── xgboost_pipeline.joblib
│   └── deployment/
│       ├── best_model.onnx
│       ├── preprocessing_schema.json
│       └── deployment_manifest.json
├── report/
│   ├── output_results.json
│   ├── saved_model_comparison_results.json
│   ├── saved_model_comparison_results.csv
│   ├── runtime_tiny_vs_xgboost.json
│   ├── runtime_tiny_vs_xgboost.csv
│   └── runtime_tiny_vs_xgboost_classification.csv
├── reports/
│   └── baseline_metrics.json
├── model.py
├── utils.py
├── curr_exp.py
├── prev_exp.py
├── compare_saved_ann_vs_non_ann.py
├── benchmark_tiny_vs_xgboost_runtime.py
└── README.md
```

## Browser Deployment Plan

The deployment app should:

1. load the selected compact ANN artifact;
2. load `preprocessing_schema.json`;
3. collect customer/account inputs from a form;
4. apply the same numeric scaling and categorical one-hot encoding used during training;
5. run inference locally in the browser;
6. display churn-risk probability and a clear business interpretation.

Target deployment approach:

- static frontend;
- GitHub Pages hosting;
- no backend;
- no API keys;
- no server-side inference;
- local browser inference using ONNX Runtime Web, TensorFlow.js, or small custom JSON weights.

## Responsible Use

This project is a portfolio and educational demonstration of lightweight browser-deployed machine learning. It is not intended for production decision making without additional validation, monitoring, security review, fairness analysis, and domain-specific compliance review.

Before any real-world use, the model should be evaluated for:

- data drift;
- calibration;
- subgroup performance and fairness;
- security of client-side artifacts;
- compliance with applicable banking, privacy, and consumer-protection rules;
- appropriate human review and escalation processes.

## Licensing Notes

Recommended split license:

- code: Apache License 2.0;
- documentation, figures, and model cards: CC BY 4.0;
- model weights: CC BY-NC 4.0 or explicit research/demo-use terms;
- dataset: Public Domain as listed by Maven Analytics for the Bank Customer Churn dataset; preserve dataset attribution in `DATASET_ATTRIBUTION.md` and `NOTICE.md`.

Recommended repository support files:

- `LICENSE`
- `NOTICE.md`
- `CITATION.cff`
- `MODEL_CARD.md`
- `DATASET_ATTRIBUTION.md`
- `RESPONSIBLE_USE.md`

## Summary

XGBoost currently provides the strongest raw predictive benchmark, but the compact ANN models are close enough to support the central deployment argument. The Tiny ANN is the strongest current deployment candidate because it accepts only a small performance hit while producing a sub-megabyte checkpoint that fits the project goal: useful, minimally sized machine learning that can run locally in a static browser app.
