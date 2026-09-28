"""CSV formula-injection mitigation: checked against the actual bytes
written by the repo's only CSV writer."""
import pandas as pd

from tabular_demand_forecast.csv_safety import sanitize_cell, write_safe_csv


def test_formula_prefixes_are_neutralised_in_file_bytes(tmp_path):
    frame = pd.DataFrame(
        {
            "text": ["=1+1", "@SUM(A1)", "+cmd", "-2+3", "\tTAB", "\rCR", "plain", "a=b"],
            "number": [-0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        }
    )
    path = tmp_path / "out.csv"
    write_safe_csv(frame, path)
    raw = path.read_bytes()
    assert b"'=1+1" in raw
    assert b"'@SUM(A1)" in raw
    assert b"'+cmd" in raw
    assert b"'-2+3" in raw
    assert b"'\tTAB" in raw
    assert b"'\rCR" in raw
    # no unprefixed formula cell starts a field
    for line in raw.split(b"\n")[1:]:
        if line:
            assert not line.startswith((b"=", b"@", b"+"))
    assert b"\nplain," in raw and b"\na=b," in raw  # benign cells untouched
    # numeric column is written as a number, not prefixed
    assert b",-0.5\n" in raw
    back = pd.read_csv(path)
    assert back["number"].tolist()[0] == -0.5


def test_sanitize_cell_leaves_non_strings():
    assert sanitize_cell(-1.0) == -1.0
    assert sanitize_cell(None) is None
    assert sanitize_cell("=x") == "'=x"


def test_header_cells_are_sanitised(tmp_path):
    path = tmp_path / "h.csv"
    write_safe_csv(pd.DataFrame({"=evil": ["x"]}), path)
    assert path.read_bytes().startswith(b"'=evil")
