from __future__ import annotations

from _bootstrap import bootstrap_local_src

bootstrap_local_src()

from soiqa_tffn.cli.inspect_one import main

if __name__ == "__main__":
    main()
