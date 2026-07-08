# Nurse LLM Scenario Tests

This folder contains manual-review benchmarks for eight nursing LLM scenarios:

1. Nursing document generation
2. Admission assessment
3. Nursing plan
4. Handoff summary
5. Discharge education
6. Chronic disease follow-up
7. Pressure injury / fall / VTE risk reminder
8. Abnormal vital signs escalation advice
9. SBAR handoff summary from raw nursing records, with professional vs simple prompts

Each scenario folder contains:

- `dataset.jsonl`: at least 10 cases. Each line includes `id`, `scenario`, `input`, and `expected_output`.
- `prompt.md`: the system prompt for that scenario.
- `README.md`: scenario-specific review notes.
- `run_compare_models.sh`: wrapper to run the four configured models for that scenario.

The `09_sbar_from_nursing_records` experiment is special: it contains `prompt_professional.md` and `prompt_simple.md`, and its script runs both prompt variants against the same dataset.

The runner does not automatically score answers. It saves model outputs next to the gold answer so humans can review quality, safety, and clinical fit.

## Run All Scenarios

```bash
cd /Users/jiaxinyu/ChangHong/projects/nurse_asr_agent/asr_llm_agent/eval_llm/nurse_llm_test
./run_compare_models.sh
```

## Run One Scenario

```bash
cd /Users/jiaxinyu/ChangHong/projects/nurse_asr_agent/asr_llm_agent/eval_llm/nurse_llm_test
./run_compare_models.sh ./01_nursing_document_generation
```

Or from inside a scenario folder:

```bash
./run_compare_models.sh
```

## Models

The default model set is:

- `Baichuan-M3`
- `Baichuan-M3-Plus`
- `qwen3.6-flash`
- `qwen3.7-plus`

Override model names with:

```bash
export BAICHUAN_M3_MODEL=Baichuan-M3
export BAICHUAN_M3_PLUS_MODEL=Baichuan-M3-Plus
export QWEN36_FLASH_MODEL=qwen3.6-flash
export QWEN37_PLUS_MODEL=qwen3.7-plus
```

## Cost Control

`BAICHUAN_WITH_SEARCH_ENHANCE` defaults to `false` to avoid automatic search billing during structured scenario evaluation.

To enable Baichuan search enhancement:

```bash
export BAICHUAN_WITH_SEARCH_ENHANCE=true
```

## Outputs

For each scenario and model:

- Results: `<scenario>/results/<model-prefix>_results.json`
- Logs: `<scenario>/logs/<model-prefix>.log`

Each result file contains:

- `summary`: model, provider, timing, parsed JSON count, token usage if returned by the provider.
- `records`: input, expected output, raw model text, parsed JSON if possible, request duration, request error if any.

## Judge Results

After model outputs are generated, run LLM-as-judge scoring:

```bash
cd /Users/jiaxinyu/ChangHong/projects/nurse_asr_agent/asr_llm_agent/eval_llm/nurse_llm_test
python judge_results.py .
```

The default judge model is `deepseek-v4-pro`.

Required environment variable:

```bash
export DEEPSEEK_API_KEY=your_key
```

Optional overrides:

```bash
export JUDGE_MODEL=deepseek-v4-pro
export JUDGE_BASE_URL=https://api.deepseek.com/v1
export JUDGE_CONCURRENCY=3
export JUDGE_LIMIT=3
```

Judge output:

- Per model result: `<scenario>/results/<model-prefix>_judge_deepseek-v4-pro.json`
- Per model CSV: `<scenario>/results/<model-prefix>_judge_deepseek-v4-pro.csv`
- Aggregate CSV: `judge_summary_deepseek-v4-pro.csv`

For prompt-variant experiments such as `09_sbar_from_nursing_records`, judge files are written inside each variant directory, for example:

```text
09_sbar_from_nursing_records/results/professional_guided/baichuan-m3_judge_deepseek-v4-pro.json
09_sbar_from_nursing_records/results/simple_guided/baichuan-m3_judge_deepseek-v4-pro.json
```

Judging dimensions:

- Faithfulness to input
- Alignment with expected output
- Clinical nursing correctness
- Safety escalation
- Task structure completeness
- Actionability and communication
- No hallucination or overreach
- JSON output quality
