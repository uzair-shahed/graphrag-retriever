# MuSiQue Dataset Directory

This document describes `dataset/musique/`, a local copy of the MuSiQue dataset release and the research code used to train, run, and evaluate several multi-hop question-answering systems.

## Purpose

MuSiQue means **Multi-hop Questions via Single-hop Question Composition**. It is a multi-hop reading-comprehension benchmark. A question is composed from several single-hop questions, and the model must use evidence from multiple Wikipedia paragraphs to answer it and identify supporting paragraphs.

The directory serves three related purposes:

1. It stores the official MuSiQue-Ans and MuSiQue-Full dataset files.
2. It provides conversion and evaluation utilities for the official and model-training formats.
3. It contains an AllenNLP-based implementation of the models and experiment configurations used in the accompanying paper.

The release also includes configurations for related 20K subsets of HotpotQA and 2WikiMultihopQA, so this directory is both a MuSiQue data release and an experiment repository.

The dataset is released under **CC BY 4.0**. See [LICENSE](../dataset/musique/LICENSE) and the upstream [README](../dataset/musique/README.md) for the original release information.

## Local data currently present

The checked-out `data/data/` directory currently contains six official JSONL splits and one leakage-prevention JSON file:

| File | Records | Description |
| --- | ---: | --- |
| `musique_ans_v1.0_train.jsonl` | 19,938 | Answerable training questions |
| `musique_ans_v1.0_dev.jsonl` | 2,417 | Answerable validation questions with labels |
| `musique_ans_v1.0_test.jsonl` | 2,459 | Answerable test questions without released labels |
| `musique_full_v1.0_train.jsonl` | 39,876 | Training questions containing answerable and unanswerable instances |
| `musique_full_v1.0_dev.jsonl` | 4,834 | Validation questions with answerability and evidence labels |
| `musique_full_v1.0_test.jsonl` | 4,918 | Test questions without released labels |
| `dev_test_singlehop_questions_v1.0.json` | 6,371 entries | Source single-hop question IDs used by MuSiQue dev/test questions |

The `data/data/` nesting is the layout present in this checkout after extraction. The public download script extracts the archive into `data/`.

### MuSiQue-Ans versus MuSiQue-Full

- **MuSiQue-Ans** contains answerable questions only. Its train/dev labels include answers and supporting paragraphs, but do not require answerability classification during the standard Ans experiments.
- **MuSiQue-Full** contains both answerable and unanswerable questions. Models must determine whether a question can be answered as well as produce an answer and supporting evidence when it is answerable.

A labeled train/dev JSONL record has these top-level fields:

- `id`: question identifier, such as `2hop__460946_294723`; the prefix indicates the hop shape.
- `question`: the composed multi-hop question.
- `question_decomposition`: the constituent single-hop questions and their answers/support paragraph indices.
- `answer`: the gold answer.
- `answer_aliases`: accepted alternative answer strings.
- `answerable`: whether the question has a valid answer.
- `paragraphs`: retrieved context paragraphs. Each paragraph has an index, title, text, and an `is_supporting` label.

A test record intentionally contains only `id`, `question`, and `paragraphs`; labels are withheld. In the checked-in examples, each instance has 20 context paragraphs. Test files must therefore be used for prediction, not local metric calculation against gold labels.

### Leakage-prevention file

`dev_test_singlehop_questions_v1.0.json` groups source question IDs under `squad2`, `natural_questions`, `zerore`, `mlqaen`, and `trex`. The file exists because MuSiQue was constructed by composing questions from these seed datasets. If a project also trains on one of those source datasets, it should exclude every source question ID listed here to avoid train/dev/test contamination.

`.answer_aliases.json` is an internal mapping from source question IDs to alternative answer strings. `raw_data_to_official_format.py` uses it to populate official `answer_aliases`, retaining aliases that occur in the supplied context.

## Directory contents

### Top-level files

- `README.md`: upstream release documentation. It covers data/prediction downloads, environment setup, model training, prediction commands, and example validation scores.
- `LICENSE`: dataset/repository license information; the README identifies the dataset license as CC BY 4.0.
- `Dockerfile`: minimal container for the leaderboard evaluator. It uses Python 3.7.4 Alpine, copies `metrics/`, and installs no training stack.
- `allennlp.diff`: small patch applied to a separately cloned AllenNLP checkout so Longformer global attention works correctly with this code.
- `run.py`: experiment entry point. It imports the local `allennlp_lib` extensions and then delegates to AllenNLP's command-line interface for commands such as `train` and `predict`. It also sets environment variables and raises the open-file limit for large runs.
- `.all_data_information.json`: mapping from raw-data paths to Google Drive IDs. It covers MuSiQue, HotpotQA-20K, and 2WikiMultihopQA-20K raw training/validation variants.
- `.all_experiment_information.json`: structured registry of downloadable experiment checkpoints and raw predictions. It records dataset, system, experiment name, prediction path, predictor name, predictor arguments, and Google Drive ID.
- `.answer_aliases.json`: answer-alias mapping described above.
- `.gitignore`: excludes downloaded/generated data, raw data, predictions, serialization directories, environments, caches, and common Python build artifacts.
- `.dockerignore`: Docker build exclusion rules.
- `download_data.sh`: downloads the official `musique_v1.0.zip` with `gdown`, extracts it, removes the archive, and removes macOS archive metadata.
- `download_predictions.sh`: downloads the official `musique_predictions_v1.0.zip` in the same way.
- `download_raw_data.py`: reads `.all_data_information.json` and downloads missing training-format raw JSONL files into `raw_data/`.
- `download_models.py`: accepts an experiment name or `all`, reads `.all_experiment_information.json`, and downloads checkpoints to `serialization_dir/<experiment_name>/model.tar.gz`.
- `download_raw_predictions.py`: accepts an experiment name or `all` and downloads registered raw prediction files under the matching `serialization_dir/<experiment_name>/predictions/` directory. It replaces path separators in source paths with `__` when constructing output names.
- `evaluate_v1.0.py`: compares official-format prediction JSONL with labeled ground truth and prints or writes metrics.
- `raw_data_to_official_format.py`: converts the model-training/raw schema to the official dataset schema, including translated IDs, paragraph indices, decomposition records, answer aliases, and optional hidden labels.
- `raw_predictions_to_official_format.py`: converts raw AllenNLP predictions into official prediction records.

The raw-data, prediction, checkpoint, and serialization directories are ignored by Git and are not currently present in this checkout. They are downloaded or generated when experiments are run.

### Official data layout

```text
data/
`-- data/
  |-- dev_test_singlehop_questions_v1.0.json
  |-- musique_ans_v1.0_{train,dev,test}.jsonl
  `-- musique_full_v1.0_{train,dev,test}.jsonl
```

### Experiment configurations

`experiment_configs/` contains JSONNet configurations. They use an external `setting` variable with `full` and `fixture` modes. `full` points to the downloaded raw data; `fixture` points to small fixture files for debugging. The configurations set readers, models, batch sizes, Longformer/RoBERTa/BART settings, optimizers, validation metrics, and random seeds.

The configuration families are:

- `end2end_model_for_*`: direct end-to-end answer and support prediction.
- `select_and_answer_model_selector_for_*`: paragraph selector/ranker.
- `select_and_answer_model_answerer_for_*`: answer/support model consuming selected paragraphs.
- `execution_model_decomposer_for_*`: BART question decomposition model. The MuSiQue decomposer is shared by the execution workflows for Ans and Full.
- `execution_by_end2end_model_for_*`: decomposer followed by an end-to-end executor.
- `execution_by_select_and_answer_model_selector_for_*`: selector for decomposed single-hop steps.
- `execution_by_select_and_answer_model_answerer_for_*`: answerer for the selected decomposed steps.

The wildcard dataset targets include:

- MuSiQue-Ans
- MuSiQue-Full
- HotpotQA-20K
- 2WikiMultihopQA-20K

The four main system designs are therefore **End2End (EE)**, **Select+Answer (SA)**, **Execution by End2End (EX(EE))**, and **Execution by Select+Answer (EX(SA))**. SA has selector and answerer components. Execution systems add a question decomposer and operate over single-hop steps.

### `allennlp_lib/`

This package contains the repository's AllenNLP extensions. `run.py` imports it before invoking AllenNLP so the custom registrations are available.

- `data/dataset_readers/`: readers for transformer reading comprehension, text ranking, and question translation, plus shared reader utilities.
- `models/`: transformer reading-comprehension, text-ranking, and BART/question-translation models and model helpers.
- `predictors/`: predictors for direct RC, in-place ranking, question translation, and multi-step EE/SA execution.
- `training/metrics/`: training-time answer, support, ranking, sufficiency, grouped, recall, SQuAD-style, and HotpotQA-style metrics.
- `nn/`: neural-network helpers.
- `tools/`: auxiliary evaluation/tooling code, including `squad.py`.

The package files present are:

```text
allennlp_lib/
|-- data/
|   `-- dataset_readers/
|       |-- question_translator.py
|       |-- text_ranker.py
|       |-- transformer_rc.py
|       `-- utils.py
|-- models/
|   |-- bart.py
|   |-- text_ranker.py
|   |-- transformer_rc.py
|   `-- utils.py
|-- nn/utils.py
|-- predictors/
|   |-- inplace_text_ranker.py
|   |-- multi_step_end2end_transformer_rc.py
|   |-- multi_step_select_and_answer_transformer_rc.py
|   |-- multi_step_transformer_rc.py
|   |-- question_translator.py
|   `-- transformer_rc.py
|-- tools/squad.py
`-- training/metrics/
  |-- answer_with_sufficiency.py
  |-- full_recall_rank.py
  |-- group.py
  |-- hotpotqa_eval.py
  |-- list_compare_em_and_f1.py
  |-- squad_em_and_f1.py
  `-- support_with_sufficiency.py
```

### `metrics/`

These are the standalone metrics used by `evaluate_v1.0.py`:

- `answer.py`: answer exact-match/F1 behavior.
- `support.py`: supporting-paragraph F1.
- `group_answer_sufficiency.py`: grouped answerability/sufficiency metric.
- `group_support_sufficiency.py`: grouped support sufficiency metric.
- `group.py` and `metric.py`: shared metric abstractions/helpers.

For answerable instances, evaluation computes answer and support metrics. If any ground-truth instance is unanswerable, evaluation additionally expects two prediction records per question and computes grouped sufficiency metrics. Predictions and gold records must have the same number of lines, and IDs must occur in the same order.

### `utils/`

- `common.py`: JSONL reading/writing, answer normalization, ID translation, and decomposition placeholder handling.
- `constants.py`: paragraph/decomposition markers, the unanswerable token, and answer-type vocabulary constants.

## Typical use

### 1. Download official data

From `dataset/musique/`:

```bash
bash download_data.sh
```

This downloads and extracts the official archive. The script installs `gdown` with `pip3` and deletes the archive afterward. Manual download is possible if Google Drive access or quota prevents `gdown` from working.

### 2. Set up the experiment environment

The original experiments used Python 3.8, AllenNLP 2.1.0, Transformers 4.7.0, `gdown` 4.5.1, and NLTK stopwords. The documented setup also clones AllenNLP v2.1.0 into the repository root, applies `allennlp.diff`, installs its dependencies, and then removes the AllenNLP package while retaining its dependencies. Longformer model weights are fetched by Transformers when needed.

The training code expects a compatible Unix-like environment and GPU configuration. The JSONNet files default to CUDA device 0 and use large pretrained models (`allenai/longformer-large-4096`, `roberta-large`, and `facebook/bart-large`).

### 3. Download raw training-format data

```bash
python download_raw_data.py
```

This creates `raw_data/` and downloads every path registered in `.all_data_information.json`. Raw files use field names expected by the custom readers and models; they are not the same serialized schema as the official files, although they represent the same underlying data.

### 4. Train a model

Set the experiment setting to `full` and invoke the local AllenNLP entry point. For example:

```bash
python run.py train experiment_configs/end2end_model_for_musique_ans_dataset.jsonnet \
  --serialization-dir serialization_dir/end2end_model_for_musique_ans_dataset
```

For a quick local/debug run, the configuration can use `fixture` and its fixture path instead of the full raw dataset. The fixture files are referenced by configurations but are not present in the current checkout.

### 5. Run prediction

Direct end-to-end prediction uses a raw input file and a predictor such as `transformer_rc`:

```bash
python run.py predict serialization_dir/end2end_model_for_musique_ans_dataset/model.tar.gz \
  raw_data/musique_ans_dev.jsonl \
  --output-file serialization_dir/end2end_model_for_musique_ans_dataset/predictions/musique_ans_dev.jsonl \
  --predictor transformer_rc --batch-size 16 --cuda-device 0 --silent
```

Select+Answer and execution workflows are sequential. A selector creates a prediction-enriched input, an answerer consumes it, and execution workflows additionally use a decomposer output. The exact paths and predictor arguments are documented in [README.md](../dataset/musique/README.md) and encoded in the corresponding JSONNet files.

To use released checkpoints instead of training:

```bash
python download_models.py end2end_model_for_musique_ans_dataset
```

Use `python download_models.py all` to download every registered checkpoint. Released raw predictions can be obtained similarly with `download_raw_predictions.py`.

### 6. Convert and evaluate predictions

Convert a raw dataset or prediction file to the official naming/schema convention:

```bash
python raw_data_to_official_format.py raw_data/musique_ans_train.jsonl
python raw_predictions_to_official_format.py raw_predictions.jsonl
```

The converters write a sibling file whose name inserts `_official_format` before the original extension. Evaluate official-format predictions against a labeled official split:

```bash
python evaluate_v1.0.py \
  predictions/musique_ans_v1.0_dev_end2end_model_predictions.jsonl \
  data/data/musique_ans_v1.0_dev.jsonl
```

The evaluator requires prediction and gold records to be aligned in both length and ID order. It reports `answer_f1` and `answer_em`, plus `support_f1`; MuSiQue-Full evaluation can also report `group_answer_sufficiency_f1` and `group_support_sufficiency_f1`.

## Important caveats

- Do not train on the released test labels: official test files do not contain labels.
- Avoid leakage from the source single-hop datasets by excluding IDs in `dev_test_singlehop_questions_v1.0.json` when using SQuAD, Natural Questions, T-REx, MLQA, or Zero Shot RE.
- Official files and raw experiment files use different field names and ID conventions. Use the supplied converters rather than manually renaming fields.
- The download scripts retrieve files from Google Drive and may require manual download if `gdown` encounters quota or confirmation problems.
- Training dependencies are old and tightly coupled: AllenNLP 2.1.0, Transformers 4.7.0, and the patched AllenNLP checkout are the versions documented by the release.
- Most experiment configurations assume a GPU, substantial memory, and access to Hugging Face pretrained weights.
- The evaluator Dockerfile is intentionally much smaller than the training environment; it is for leaderboard-style evaluation only.

## Upstream reference

The authoritative project README and paper links are maintained in [dataset/musique/README.md](../dataset/musique/README.md). This description documents the contents and behavior observed in the local checkout, including the locally available record counts and directory layout.
