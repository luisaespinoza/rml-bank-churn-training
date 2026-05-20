# Model Card: Bank Customer Churn Tiny ANN

## Model Overview

**Model name:** Tiny ANN Bank Customer Churn Scorer  
**Model type:** Compact artificial neural network for binary classification  
**Primary task:** Predict whether a bank customer is likely to churn  
**Target variable:** `Exited`  
**Current deployment candidate:** Tiny ANN  
**Framework:** PyTorch  
**Intended deployment:** Static browser application with local inference

This model card documents the current deployment candidate for the Bank Customer Churn Risk Scorer project. The project demonstrates a minimal deployable ML tradeoff: accept a small performance loss versus the strongest tabular benchmark in exchange for a smaller, simpler model that can run locally in a browser without backend inference, API keys, or paid cloud services.

## Intended Use

The model is intended for portfolio, educational, and demonstration use. It estimates churn risk from structured customer/account attributes and can support a hypothetical customer-retention prioritization workflow.

Example use case:

- rank customers by predicted churn risk;
- identify the highest-risk 20% of customers;
- demonstrate how a lightweight model can support retention-targeting decisions;
- show that useful ML systems do not need to be chatbots, GenAI products, or cloud-hosted inference services.

## Out-of-Scope Uses

This model is **not** intended for production banking, credit, eligibility, pricing, underwriting, or automated customer treatment decisions.

Do not use this model for real-world decision making without additional validation, monitoring, security review, fairness analysis, privacy review, and domain-specific compliance review.

## Input Features

Expected features include:

| Feature | Description |
|---|---|
| `CreditScore` | Customer credit score |
| `Geography` | Customer geography/category |
| `Gender` | Customer gender/category |
| `Age` | Customer age |
| `Tenure` | Customer tenure with the bank |
| `Balance` | Account balance |
| `NumOfProducts` | Number of bank products used |
| `HasCrCard` | Credit-card indicator |
| `IsActiveMember` | Active membership indicator |
| `EstimatedSalary` | Estimated salary |

Identifier columns are dropped if present:

- `RowNumber`
- `CustomerId`
- `Surname`

## Preprocessing

The training pipeline applies:

- `StandardScaler` to numeric columns;
- `OneHotEncoder(handle_unknown="ignore")` to categorical columns;
- preprocessing fit on the training set only;
- validation/test data transformed with the fitted preprocessing pipeline.

For browser deployment, preprocessing metadata is exported to:

```text
artifacts/deployment/preprocessing_schema.json
```

This schema stores numeric scaling parameters and categorical one-hot category ordering so that the browser application can reproduce the training-time feature transformation.

## Architecture

The current deployment candidate is the **Tiny ANN**:

```text
input -> 32 -> 16 -> 1
```

Characteristics:

| Property | Value |
|---|---:|
| Input dimension after preprocessing | 13 |
| Parameter count | 993 |
| Estimated FP32 parameter size | 0.003788 MB |
| Saved checkpoint size | 0.339579 MB |
| Hidden activations | ReLU |
| Output | Single binary-classification logit |
| Prediction transform | Sigmoid |

## Training Setup

Canonical current run:

| Setting | Value |
|---|---:|
| Train/validation/test split | 70% / 15% / 15% |
| Stratification | By `Exited` |
| Seed | 314 |
| Epochs | 15,000 |
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Loss | `BCEWithLogitsLoss` |
| Checkpoint selection | Best validation loss |

## Evaluation Metrics

The project reports both general ML metrics and business-targeting metrics:

| Metric | Meaning |
|---|---|
| ROC-AUC | Overall ranking quality across thresholds |
| PR-AUC / Average Precision | Precision-recall quality for churn prediction |
| Recall@Top 20% | Fraction of actual churners captured among the highest-risk 20% of customers |
| Precision@Top 20% | Fraction of the highest-risk 20% who actually churned |
| Brier Score | Probability calibration error; lower is better |
| Log Loss | Probabilistic classification loss; lower is better |

## Current Tiny ANN Performance

Seed 314 / 15k epoch checkpoint result:

| Metric | Value |
|---|---:|
| ROC-AUC | 0.866515 |
| PR-AUC | 0.718818 |
| Recall@Top 20% | 0.622951 |
| Precision@Top 20% | 0.633333 |
| Brier Score | 0.099371 |
| Log Loss | 0.328620 |

## Benchmark Context

The strongest tabular benchmark in the current seed-314 runtime comparison is XGBoost:

| Model | ROC-AUC | PR-AUC | Recall@Top 20% | Precision@Top 20% |
|---|---:|---:|---:|---:|
| XGBoost | 0.880337 | 0.743483 | 0.655738 | 0.666667 |
| Tiny ANN | 0.866515 | 0.718818 | 0.622951 | 0.633333 |

Compared with XGBoost, the Tiny ANN gives up approximately:

- 0.013823 ROC-AUC;
- 3.3 percentage points of Recall@Top 20%;
- 3.3 percentage points of Precision@Top 20%.

This performance gap is accepted deliberately in exchange for a much smaller and simpler browser-deployable model.

## Runtime Context

Local Python CPU runtime benchmark, median model-only inference:

| Batch Size | Tiny ANN Median | XGBoost Median | Tiny Speedup |
|---:|---:|---:|---:|
| 1 | 0.062 ms | 0.320 ms | 5.1x |
| 32 | 0.067 ms | 0.468 ms | 7.0x |
| 300 | 0.087 ms | 1.484 ms | 17.2x |
| 1500 | 0.153 ms | 6.042 ms | 39.5x |

These are local Python CPU timings, not browser timings. Browser runtime should be measured separately in the deployed web app.

## Limitations

Known limitations:

- evaluated on a single public churn dataset variant;
- not validated on real bank production data;
- no fairness or subgroup analysis has been completed yet;
- no long-term drift monitoring exists;
- client-side deployment exposes model artifacts to users;
- metrics are based on offline evaluation and do not prove business impact;
- calibration may need additional evaluation before any decision-support use.

## Ethical and Responsible-Use Considerations

Churn-risk prediction can affect customer treatment and resource allocation. Before any real-world use, the model should be evaluated for:

- subgroup performance and fairness;
- calibration and uncertainty;
- data drift;
- privacy and security risks;
- explainability needs for stakeholders;
- compliance with applicable banking, privacy, and consumer-protection rules;
- appropriate human review and escalation processes.

## Deployment Decision

The current deployment candidate is **Tiny ANN**.

This choice is not based on Tiny being the strongest raw predictive model. XGBoost remains the stronger tabular benchmark. Tiny is selected because it better matches the project goal: useful, minimally sized machine learning that can run locally in a static browser application.

## Artifacts

Relevant artifacts:

```text
artifacts/models/tiny.pt
artifacts/models/best_model.pt
artifacts/deployment/preprocessing_schema.json
artifacts/deployment/deployment_manifest.json
report/output_results.json
report/runtime_tiny_vs_xgboost.json
```

## Versioning Notes

Current documented result set:

- seed: 314;
- ANN epochs: 15,000;
- deployment candidate: Tiny ANN;
- comparison benchmark: XGBoost;
- older seed-42 exploratory results are not used for the current deployment decision.
