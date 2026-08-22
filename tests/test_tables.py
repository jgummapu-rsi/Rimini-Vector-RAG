from app.pipeline.tables import (
    df_to_markdown,
    looks_like_markdown_table,
    rows_to_markdown,
)


def test_rows_to_markdown_basic():
    md = rows_to_markdown([["a", "b"], ["1", "2"]])
    lines = md.splitlines()
    assert lines[0] == "| a | b |"
    assert lines[1] == "| --- | --- |"
    assert lines[2] == "| 1 | 2 |"


def test_rows_to_markdown_escapes_pipes_and_newlines():
    md = rows_to_markdown([["h"], ["a|b\nc"]])
    assert "\\|" in md
    assert "\n" not in md.splitlines()[2]  # newline inside cell flattened


def test_rows_to_markdown_pads_ragged_rows():
    md = rows_to_markdown([["a", "b", "c"], ["1"]])
    assert md.splitlines()[2] == "| 1 |  |  |"


def test_df_to_markdown_uses_columns_as_header():
    import pandas as pd
    df = pd.DataFrame({"Region": ["APAC"], "Rev": [120]})
    md = df_to_markdown(df)
    assert md.splitlines()[0] == "| Region | Rev |"
    assert "| APAC | 120 |" in md


def test_looks_like_markdown_table():
    assert looks_like_markdown_table("| a | b |\n| --- | --- |\n| 1 | 2 |")
    assert not looks_like_markdown_table("just some text")
    assert not looks_like_markdown_table("| a | b |")  # no separator row
