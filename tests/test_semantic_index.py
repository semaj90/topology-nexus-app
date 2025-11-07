import json
from pathlib import Path

from semantic.index_builder import build_semantic_index


def test_semantic_index_builder(tmp_path):
    dataset = Path('docs/llms/crawled/ts_drizzle_uno.jsonl')
    output = tmp_path / 'semantic_index.json'
    count = build_semantic_index(dataset, output)

    assert count > 0
    assert output.exists()

    data = json.loads(output.read_text(encoding='utf-8'))
    assert isinstance(data, list)
    assert all('tags' in item for item in data)
