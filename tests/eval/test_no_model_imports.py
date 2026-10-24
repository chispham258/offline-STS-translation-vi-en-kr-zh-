"""D5: the evaluator must not depend on pipeline or model code."""

import subprocess
import sys

FORBIDDEN = (
    "torch",
    "transformers",
    "onevoice.pipeline",
    "onevoice.components",
    "onnxruntime",
    "ctranslate2",
    "faster_whisper",
    "sentencepiece",
)


CHECK = """
import sys
import onevoice.eval.evaluator  # noqa: F401
forbidden = {forbidden!r}
print(",".join(sorted(set(sys.modules) & set(forbidden))))
"""


def test_importing_the_evaluator_pulls_in_no_model_code():
    result = subprocess.run(
        [sys.executable, "-c", CHECK.format(forbidden=FORBIDDEN)],
        capture_output=True,
        text=True,
        check=True,
    )
    leaked = [name for name in result.stdout.strip().split(",") if name]
    assert not leaked, f"evaluator imported model code: {leaked}"
