# Whitelists and safe loading
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