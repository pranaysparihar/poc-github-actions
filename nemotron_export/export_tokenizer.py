from pathlib import Path
from huggingface_hub import hf_hub_download

repo = "nvidia/Nemotron-3-Embed-1B-BF16"
out = Path("tokenizer_export")
out.mkdir(exist_ok=True)
for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
    try:
        p = Path(hf_hub_download(repo_id=repo, filename=name))
    except Exception as exc:
        print(f"skip {name}: {exc}")
        continue
    (out / name).write_bytes(p.read_bytes())
    print(name, p.stat().st_size)
