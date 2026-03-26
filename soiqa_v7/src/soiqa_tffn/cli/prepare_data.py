from __future__ import annotations

from soiqa_tffn.cli.build_manifest import main as build_manifest_main
from soiqa_tffn.cli.split_manifest import main as split_manifest_main


def main() -> None:
    build_manifest_main()
    split_manifest_main()


if __name__ == "__main__":
    main()
