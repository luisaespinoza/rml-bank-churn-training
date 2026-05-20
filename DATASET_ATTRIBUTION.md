# Dataset Attribution Note

## Dataset

This project uses the **Bank Customer Churn** dataset from Maven Analytics Data Playground.

The dataset contains account information for 10,000 customers at a European bank and is provided by Maven Analytics as a free sample/practice dataset. Maven lists the dataset as a single-table CSV with 10,000 records and 13 fields.

The binary target column is:

```text
Exited
```

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

Identifier columns are dropped during preprocessing if present:

- `RowNumber`
- `CustomerId`
- `Surname`

## Source

```text
Dataset: Bank Customer Churn
Provider: Maven Analytics
Collection: Maven Analytics Data Playground
Source page: https://mavenanalytics.io/data-playground/bank-customer-churn
Direct download: https://maven-datasets.s3.amazonaws.com/Bank+Customer+Churn/Bank+Customer+Churn.zip
Access date: 2026-05-19
```

## License and Terms

Maven Analytics lists this dataset's source as Kaggle and its license as Public Domain on the Bank Customer Churn Data Playground page.

The dataset is still third-party data, and this repository does not claim ownership of it. The repository license applies to this project's code and documentation, not to the third-party dataset.
## Redistribution Guidance

To avoid licensing ambiguity, this repository should not redistribute the raw dataset unless Maven Analytics' terms clearly permit redistribution.

Recommended repository practice:

- do not commit `data/Bank_Churn.csv` to the public repository unless redistribution is confirmed;
- provide the source page and direct download link instead;
- instruct users to download the dataset from Maven Analytics;
- document the expected local file path:

```text
data/Bank_Churn.csv
```

Suggested README language:

```md
The raw dataset is not included in this repository. Download the Bank Customer Churn dataset from Maven Analytics Data Playground, then place the CSV at `data/Bank_Churn.csv` before running the training scripts.
```

## Dataset Handling in This Project

The training pipeline performs the following dataset handling steps:

1. loads the CSV from:

```text
data/Bank_Churn.csv
```

2. drops identifier columns if present:

```text
RowNumber, CustomerId, Surname
```

3. splits the data into train, validation, and test sets:

```text
70% train / 15% validation / 15% test
```

4. stratifies the split by the target column:

```text
Exited
```

5. fits preprocessing only on the training split;
6. transforms validation and test splits with the fitted preprocessing pipeline.

## Known Dataset Limitations

The following limitations should be considered when interpreting model results:

- the dataset is a sample/practice dataset, not a production banking dataset;
- the source page does not provide a full data-generation or collection methodology;
- the dataset may not represent current banking populations, policies, or customer behavior;
- the churn label definition may be simplified relative to real retention workflows;
- demographic and geography fields may not be sufficient for production fairness analysis;
- performance on this dataset should not be treated as evidence of production readiness.

## Citation

This project cites the dataset as follows:

```text
Maven Analytics. Bank Customer Churn dataset. Maven Analytics Data Playground.
Source page: https://mavenanalytics.io/data-playground/bank-customer-churn
Direct download: https://maven-datasets.s3.amazonaws.com/Bank+Customer+Churn/Bank+Customer+Churn.zip
Original source listed by Maven: Kaggle
License listed by Maven: Public Domain
Accessed: 2026-05-19.
```
