from pathlib import Path
import sys

import jsonlines

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qlora.dataset_builder import DatasetBuilder


def test_dataset_builder_creates_chunks(tmp_path: Path):
    doc = tmp_path / "sample.txt"
    doc.write_text("Paragraph one.\n\nParagraph two is a bit longer to ensure chunking works.", encoding="utf-8")

    builder = DatasetBuilder(chunk_size=32, overlap=4)
    output_path = tmp_path / "dataset.jsonl"
    builder.build_jsonl([doc], output_path)

    with jsonlines.open(output_path) as reader:
        rows = list(reader)

    assert rows, "Dataset should contain at least one chunk"
    assert rows[0]["text"].strip(), "Chunk text should not be empty"
