"""Regression test for .json parsing in unsloth/dataprep/raw_text.py.

Both .json and .jsonl map to the "json_lines" handler, which used to parse the
file one line at a time. A real .json file is a single JSON document (commonly
a top-level list of records), so every line failed json.loads, the whole
document was dropped, and the handler returned "" (load_from_file then rejected
the valid file as "empty"). The handler now parses the file as one JSON value
first and falls back to line-by-line for true .jsonl.

raw_text.py's only third-party import is `datasets`, so we stub it and exec the
module directly, with no `import unsloth` (which needs a GPU / unsloth_zoo).
"""

import json
import sys
import types
from pathlib import Path

RAW_TEXT_PATH = Path(__file__).parents[1] / "unsloth" / "dataprep" / "raw_text.py"


def _load_raw_text():
    sys.modules.setdefault("datasets", types.SimpleNamespace(Dataset = object))
    module = types.ModuleType("unsloth_raw_text_under_test")
    exec(
        compile(RAW_TEXT_PATH.read_text(encoding = "utf-8"), str(RAW_TEXT_PATH), "exec"),
        module.__dict__,
    )
    return module


def test_json_document_is_parsed_whole(tmp_path):
    loader = _load_raw_text().RawTextDataLoader(tokenizer = object())
    path = tmp_path / "data.json"
    path.write_text(
        json.dumps([{"text": "hello world"}, {"text": "second sample"}], indent = 2), encoding = "utf-8"
    )
    assert loader._read_file_by_format(str(path), "json_lines") == "hello world\n\nsecond sample"


def test_jsonl_is_still_parsed_line_by_line(tmp_path):
    loader = _load_raw_text().RawTextDataLoader(tokenizer = object())
    path = tmp_path / "data.jsonl"
    path.write_text('{"text": "a"}\n{"text": "b"}\n', encoding = "utf-8")
    assert loader._read_file_by_format(str(path), "json_lines") == "a\n\nb"
