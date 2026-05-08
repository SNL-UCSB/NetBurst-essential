"""
Deprecated entrypoint for the NetBurst pipeline.

Use instead (from ``src/train/``):

- ``python pretrain_twin.py ...`` — cold-start training
- ``python finetune_twin.py ...`` — resume from a NetBurst checkpoint dir
- ``python ar_predict.py ...`` — distributed AR inference
- ``python extract_repr.py {final|detailed|twin_ip} ...`` — representations
- ``python finetune_repr_mlp.py ...`` — MLP on saved representation CSVs
"""

from __future__ import annotations

import argparse
import sys
import warnings


def main() -> None:
    warnings.warn(
        "pretrain_netburst_head_2.py is deprecated; use pretrain_twin.py, finetune_twin.py, "
        "ar_predict.py, extract_repr.py, or finetune_repr_mlp.py",
        DeprecationWarning,
        stacklevel=1,
    )
    if len(sys.argv) < 2:
        print(
            "Usage: python pretrain_netburst_head_2.py "
            "{main|inference_auto_regression|get_final_representations|get_detailed_representations|get_representations_with_ip} ...",
            file=sys.stderr,
        )
        sys.exit(2)

    cmd = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if cmd == "main":
        pre = argparse.ArgumentParser(add_help=False)
        pre.add_argument("--retrain", type=str, default=None)
        known, _ = pre.parse_known_args()
        from netburst.train_loop import run_training

        run_training(from_checkpoint=known.retrain is not None)
    elif cmd == "inference_auto_regression":
        from ar_predict import main as ar_main

        ar_main()
    elif cmd == "get_final_representations":
        sys.argv.insert(1, "final")
        from extract_repr import main as er_main

        er_main()
    elif cmd == "get_detailed_representations":
        sys.argv.insert(1, "detailed")
        from extract_repr import main as er_main

        er_main()
    elif cmd == "get_representations_with_ip":
        sys.argv.insert(1, "twin_ip")
        from extract_repr import main as er_main

        er_main()
    else:
        print(f"Unknown subcommand {cmd!r}. See module docstring for replacements.", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
