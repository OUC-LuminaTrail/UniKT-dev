# Data Processing

Download, preprocess, and split datasets for knowledge tracing experiments.

## Pipeline Overview

```mermaid
flowchart LR
    A[Raw Data] --> B[Download]
    B --> C[Clean]
    C --> D[Serialize]
    D --> E[K-Fold Split]
```

## Supported Datasets

| Dataset | Source |
|---------|--------|
| `assistments09` | [ASSISTmentsData](https://sites.google.com/site/assistmentsdata/datasets/2009-2010-assistment-data) |
| `assistments12` | [ASSISTmentsData](https://sites.google.com/site/assistmentsdata/datasets/2012-13-school-data-with-affect) |
| `assistments17` | [ASSISTmentsData](https://sites.google.com/site/assistmentsdata/datasets/2017-assistments-data) |
| `ednet_kt1` | [GitHub](https://github.com/riiid/ednet) |

## Quick Start

### Download Data

```bash
# Download from source
python data_process.py download -d assistments09

# Force re-download
python data_process.py download -d assistments09 --force

# Custom download URL
python data_process.py download -d assistments09 --data_url https://example.com/data.zip
```

**Download Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--force` | False | Force re-download even if file exists |
| `--max_retries` | 3 | Maximum number of download retries |
| `--num_threads` | 4 | Number of threads for parallel download |
| `--data_url` | None | Override data URL for downloading |

### Process Data

```bash
# Process with default settings
python data_process.py process -d assistments09

# Custom parameters
python data_process.py process \
    -d assistments09 \
    --min_seq_len 3 \
    --max_seq_len 200 \
    --kfold 5 \
    --test_ratio 0.2 \
    --seed 42
```

**Processing Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--min_seq_len` | 3 | Minimum sequence length |
| `--max_seq_len` | 200 | Maximum sequence length |
| `--kfold` | 5 | Number of folds for K-Fold cross-validation |
| `--test_ratio` | 0.2 | Ratio for test set |
| `--seed` | 42 | Random seed for reproducibility |

**Sampling Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--sample_users` | None | Number of users to sample (None to disable) |
| `--sample_strategy` | random | Sampling strategy (random, stratified) |
| `--sample_attempts_bins` | 20 100 | Attempt count bin edges for stratified sampling |
| `--sample_correct_bins` | 0.4 0.8 | Correct rate bin edges for stratified sampling |

**Extra Processing:**

| Parameter | Description |
|-----------|-------------|
| `--extra windowslate` | Build windowlate data for sliding window training |

## Output Structure

After processing, the following files are generated in `data/{dataset}/`:

```
data/{dataset}/
├── {dataset}_question.parquet       # Question features
├── {dataset}_sequence.parquet       # User interaction sequences
├── {dataset}_split_question_sequence.parquet  # Question interaction sequences by fold
├── {dataset}_split_skill_sequence.parquet     # Skill interaction sequences by fold
├── {dataset}_windowlate.parquet     # (Optional) Windowlate data
└── metadata.json                    # Processing metadata
```

### File Descriptions

| File | Description |
|------|-------------|
| `*_question.parquet` | Question features (question_id, skill, assignment, etc.) |
| `*_sequence.parquet` | User interaction sequences |
| `*_split_question_sequence.parquet` | Question interaction sequences split by K-fold |
| `*_split_skill_sequence.parquet` | Skill interaction sequences split by K-fold |
| `*_windowlate.parquet` | Pre-windowed sequences for sliding window training |
| `metadata.json` | Processing parameters and file checksums |

> [!NOTE]
> **Question Sequence vs Skill Sequence**
>
> | Type | Granularity | Use Case |
> |------|-------------|----------|
> | **Question Sequence** | One interaction = one timestep | Predict performance on specific questions |
> | **Skill Sequence** | One skill = one timestep | Predict skill mastery (multi-skill questions expanded) |
>
> For multi-skill questions (e.g., a question tagged with skills `2`, `37`, `70`):
> - **Question sequence**: Single entry with `question_id=123`
> - **Skill sequence**: Three entries with `skill_id=[2, 37, 70]`, same `label` and other fields
>
> The skill sequence is longer than the question sequence when multi-skill questions exist.

### Sequence Data Fields

Each sequence file contains the following fields:

| Field | Shape | Description |
|-------|-------|-------------|
| `user` | `[B]` | User identifiers |
| `question_id` | `[B, S]` | Question IDs |
| `response` | `[B, S]` | Correctness (0/1) |
| `mask` | `[B, S]` | Valid position mask |
| `fold` | `[B]` | Fold label for K-Fold cross-validation |

## Column Mapping

Each dataset has different raw column names. The following tables show how raw columns are mapped to standardized names during processing.

### Standard Output Columns

| Column | Description |
|--------|-------------|
| `user` | User identifier (remapped to consecutive integers) |
| `question` | Question identifier (remapped to consecutive integers) |
| `skill` | Skill/concept identifier (remapped to consecutive integers) |
| `assignment` | Assignment/task identifier |
| `template` | Question template identifier |
| `label` | Response correctness (0 or 1) |
| `attempt_count` | Number of attempts on this question |
| `hint_count` | Number of hints used |
| `timestamp` | Unix timestamp (milliseconds) or sequential order |
| `fold` | K-fold label (-1 for test, 0 to n_splits-1 for train/val) |

### ASSISTments 2009

| Raw Column | Standard Column |
|------------|-----------------|
| `user_id` | `user` |
| `problem_id` | `question` |
| `correct` | `label` |
| `skill_id` | `skill` |
| `assignment_id` | `assignment` |
| `template_id` | `template` |
| `order_id` | `timestamp` |

**Notes:**
- Skills are split by `_` (e.g., `"2_37_70"` → `["2", "37", "70"]`)
- Multi-skill questions are expanded into multiple rows in the question data.

### ASSISTments 2012

| Raw Column | Standard Column |
|------------|-----------------|
| `user_id` | `user` |
| `problem_id` | `question` |
| `correct` | `label` |
| `skill_id` | `skill` |
| `assignment_id` | `assignment` |
| `template_id` | `template` |
| `start_time` | `timestamp` |

**Notes:**
- `start_time` is parsed as datetime and converted to Unix milliseconds
- Rows with null skills are filtered out

### ASSISTments 2017

| Raw Column | Standard Column |
|------------|-----------------|
| `studentId` | `user` |
| `problemId` | `question` |
| `correct` | `label` |
| `skill` | `skill` |
| `assignmentId` | `assignment` |
| `hintCount` | `hint_count` |
| `attemptCount` | `attempt_count` |
| `startTime` | `timestamp` |

**Notes:**
- No template column in this dataset
- Rows with null skills are filtered out

### EdNet-KT1

| Raw Column | Standard Column |
|------------|-----------------|
| `user_id` (from filename) | `user` |
| `question_id` | `question` |
| `user_answer` + `correct_answer` | `label` (computed) |
| `timestamp` | `timestamp` |
| `tags` | `skill` and `template` |
| `bundle_id` | `assignment` |

**Notes:**
- `label` is computed by comparing `user_answer` with `correct_answer`
- `attempt_count` and `hint_count` are placeholders (set to 1 and 0)
- Tags are split by `;` (e.g., `"algebra;geometry"` → `["algebra", "geometry"]`)
- Tags are deduplicated and sorted before splitting

## Adding Custom Datasets

### Step 1: Create Processor

```python
# utils/data_process/my_dataset.py
from utils.data_process.base import DataSource

class MyDatasetProcessor(DataSource):
    def download(self):
        """Download raw data"""
        pass

    def process(self):
        """Clean and serialize data"""
        pass
```

### Step 2: Register

```python
# utils/data_process/data_source.py
DATA_SOURCES.register("my_dataset", "utils.data_process.my_dataset", "MyDatasetProcessor")
```

### Step 3: Run

```bash
python data_process.py process -d my_dataset
```

## Notes

- **Data Leakage**: K-fold split preserves temporal order within sequences
- **Memory**: Large datasets (EdNet) require significant RAM during processing
- **Idempotency**: Re-running `process` overwrites previous results

## Related Docs

- [Quick Start](quick_start.md) - Installation
- [Training](training.md) - Use processed data
