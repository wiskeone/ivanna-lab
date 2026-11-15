"""Curated benchmark catalog: the popular top list shown on the Benchmarks page.

Each entry maps to an lm-eval-harness task id (https://github.com/EleutherAI/
lm-evaluation-harness). The full registry (600+ tasks) is synced separately
from TaskManager introspection (refresh_benchmark_catalog_task); curated rows
are authoritative for descriptions/metrics/categories and are never
overwritten by the registry sync.
"""

CURATED_BENCHMARKS = [
    {
        'name': 'mmlu',
        'display_name': 'MMLU',
        'category': 'knowledge',
        'group': 'mmlu',
        'metrics': ['acc', 'acc_stderr'],
        'dataset_path': 'cais/mmlu',
        'dataset_name': 'all',
        'n_samples': 14042,
        'popular': True,
        'description': 'Massive Multitask Language Understanding: 57 subjects from '
                       'STEM to humanities, multiple-choice (4 options). The classic '
                       'knowledge benchmark; full run is long — use a limit to start.',
    },
    {
        'name': 'gsm8k',
        'display_name': 'GSM8K',
        'category': 'math',
        'group': 'gsm8k',
        'metrics': ['exact_match'],
        'dataset_path': 'openai/gsm8k',
        'dataset_name': 'main',
        'n_samples': 1319,
        'popular': True,
        'description': 'Grade-school math word problems requiring multi-step '
                       'arithmetic reasoning. The standard math-reasoning benchmark.',
    },
    {
        'name': 'hellaswag',
        'display_name': 'HellaSwag',
        'category': 'reasoning',
        'group': 'hellaswag',
        'metrics': ['acc', 'acc_norm'],
        'dataset_path': 'Rowan/hellaswag',
        'n_samples': 10042,
        'popular': True,
        'description': 'Commonsense sentence completion: pick the most plausible '
                       'continuation of a short scenario. Adversarially filtered.',
    },
    {
        'name': 'arc_challenge',
        'display_name': 'ARC-Challenge',
        'category': 'reasoning',
        'group': 'arc',
        'metrics': ['acc', 'acc_norm'],
        'dataset_path': 'allenai/ai2_arc',
        'dataset_name': 'ARC-Challenge',
        'n_samples': 1172,
        'popular': True,
        'description': 'AI2 Reasoning Challenge, the challenge split: grade-school '
                       'science questions that defeat retrieval and co-occurrence methods.',
    },
    {
        'name': 'winogrande',
        'display_name': 'WinoGrande',
        'category': 'reasoning',
        'group': 'winogrande',
        'metrics': ['acc'],
        'dataset_path': 'allenai/winogrande',
        'dataset_name': 'winogrande_xl',
        'n_samples': 1767,
        'popular': True,
        'description': 'Pronoun resolution over adversarially filtered Winograd-style '
                       'schemata; tests commonsense coreference.',
    },
    {
        'name': 'truthfulqa_mc',
        'display_name': 'TruthfulQA (MC)',
        'category': 'safety',
        'group': 'truthfulqa',
        'metrics': ['acc', 'acc_norm'],
        'dataset_path': 'truthfulqa/truthful_qa',
        'dataset_name': 'multiple_choice',
        'n_samples': 817,
        'popular': True,
        'description': 'Questions humans typically answer falsely because of common '
                       'misconceptions; measures truthfulness under imitative falsehoods.',
    },
    {
        'name': 'ifeval',
        'display_name': 'IFEval',
        'category': 'instruction',
        'group': 'ifeval',
        'metrics': ['prompt_level_strict_acc', 'inst_level_strict_acc'],
        'dataset_path': 'google/IFEval',
        'n_samples': 541,
        'popular': True,
        'description': 'Instruction-following with verifiable constraints (write N '
                       'words, include keywords, format rules). Strict programmatic checks.',
    },
    {
        'name': 'gpqa_main',
        'display_name': 'GPQA (main)',
        'category': 'knowledge',
        'group': 'gpqa',
        'metrics': ['acc_norm'],
        'dataset_path': 'Idavidrein/gpqa',
        'dataset_name': 'gpqa_main',
        'n_samples': 198,
        'popular': True,
        'description': 'Graduate-level Google-proof science questions written by domain '
                       'experts. Hard even for strong models. Requires dataset access '
                       'acceptance on the HF Hub.',
    },
    {
        'name': 'bbh_cot_fewshot',
        'display_name': 'BIG-Bench Hard (CoT)',
        'category': 'reasoning',
        'group': 'bbh',
        'metrics': ['acc_norm'],
        'n_samples': 6511,
        'popular': True,
        'description': '27 challenging BIG-Bench tasks with chain-of-thought few-shot '
                       'prompting; tasks where earlier models lagged the human average.',
    },
    {
        'name': 'math_algebra',
        'display_name': 'MATH (algebra subset)',
        'category': 'math',
        'group': 'math',
        'metrics': ['exact_match'],
        'dataset_path': 'EleutherAI/hendrycks_math',
        'dataset_name': 'algebra',
        'n_samples': 1187,
        'popular': True,
        'description': 'Competition math (Hendrycks MATH), algebra subset with exact-match '
                       'grading. The full MATH set is much longer — subsets are the practical start.',
    },
    {
        'name': 'mgsm_en',
        'display_name': 'MGSM (English)',
        'category': 'multilingual',
        'group': 'mgsm',
        'metrics': ['exact_match'],
        'dataset_path': 'juletxara/mgsm',
        'dataset_name': 'en',
        'n_samples': 250,
        'popular': True,
        'description': 'Multilingual Grade School Math (translated GSM8K sample). The '
                       'entry point of the multilingual math suite (mgsm_* tasks).',
    },
    {
        'name': 'humaneval',
        'display_name': 'HumanEval',
        'category': 'code',
        'group': 'humaneval',
        'metrics': ['pass@1'],
        'dataset_path': 'openai/openai_humaneval',
        'n_samples': 164,
        'popular': True,
        'description': 'Hand-written Python programming problems scored by unit-test '
                       'execution (pass@k). Requires code generation in pass@1 mode.',
    },
    {
        'name': 'triviaqa',
        'display_name': 'TriviaQA',
        'category': 'knowledge',
        'group': 'triviaqa',
        'metrics': ['exact_match'],
        'dataset_path': 'mandarjoshi/trivia_qa',
        'dataset_name': 'rc.nocontext',
        'n_samples': 17944,
        'popular': True,
        'description': 'Open-domain trivia with authored question-answer pairs; a '
                       'closed-book factual knowledge probe. Long dataset — use a limit.',
    },
    {
        'name': 'piqa',
        'display_name': 'PIQA',
        'category': 'reasoning',
        'group': 'piqa',
        'metrics': ['acc', 'acc_norm'],
        'dataset_path': 'ybisk/piqa',
        'n_samples': 1838,
        'popular': True,
        'description': 'Physical interaction QA: choosing the sensible way to accomplish '
                       'everyday physical tasks.',
    },
    {
        'name': 'boolq',
        'display_name': 'BoolQ',
        'category': 'knowledge',
        'group': 'boolq',
        'metrics': ['acc'],
        'dataset_path': 'google/boolq',
        'n_samples': 3270,
        'popular': True,
        'description': 'Naturally occurring yes/no reading-comprehension questions '
                       'paired with Wikipedia passages.',
    },
]


def curated_entries():
    """The curated catalog (list of dicts)."""
    return CURATED_BENCHMARKS
