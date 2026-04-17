from __future__ import annotations

import sys
from pathlib import Path

import yaml


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: configure_birdnet_url.py CONFIG_PATH BIRDNET_URL", file=sys.stderr)
        return 2

    config_path = Path(sys.argv[1])
    birdnet_url = sys.argv[2]
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    classifier = data.setdefault("classifier", {})
    if classifier.get("birdnet_api_url"):
        print("BirdNET URL already configured.")
        return 0

    classifier["birdnet_api_url"] = birdnet_url
    config_path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"BirdNET URL set to {birdnet_url}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
