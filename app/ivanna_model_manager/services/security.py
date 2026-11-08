# Белые списки и безопасная загрузка
ALLOWED_HUGGINGFACE_PATTERNS = [
    "Qwen/*",
    "meta-llama/*",
    "mistralai/*",
    "google/*",
    "bigscience/*",
    "databricks/*",
    "microsoft/*",
]

BLACKLISTED_PATTERNS = [
    "*porn*",
    "*hate*",
    "*violence*",
    "*illegal*",
]