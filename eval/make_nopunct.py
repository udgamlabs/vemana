# Copyright 2026 Udgam Labs. Licensed under the Apache License, Version 2.0 (https://www.apache.org/licenses/LICENSE-2.0).
"""Diagnostic variant of the Telugu fresh-text set: ASCII punctuation removed and every whitespace run (newlines too)
collapsed to one space. nanochat-telugu-560M's greedy output has no punctuation or line breaks, which suggests its training text
had them stripped; scoring every model on this copy shows how much of a bits-per-byte gap is punctuation alone.
Not a headline benchmark: the headline is the unmodified set.

  .venv/bin/python fresh_text/make_nopunct.py   -> fresh_text/te-2026-10/data/test_nopunct.parquet
"""
import os
import re
import string

import pyarrow as pa
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
src = os.path.join(HERE, "te-2026-10", "data", "test.parquet")
t = pq.read_table(src, columns=["id", "text"])
punct = re.compile("[" + re.escape(string.punctuation) + "–—‘’“”…]")
texts = [re.sub(r"\s+", " ", punct.sub(" ", x)).strip() for x in t.column("text").to_pylist()]
out = os.path.join(HERE, "te-2026-10", "data", "test_nopunct.parquet")
pq.write_table(pa.table({"id": t.column("id"), "text": texts}), out)
b0 = sum(len(x.encode()) for x in t.column("text").to_pylist())
b1 = sum(len(x.encode()) for x in texts)
print(f"{len(texts)} docs, {b0 / 1e6:.2f} MB -> {b1 / 1e6:.2f} MB ({100 * (1 - b1 / b0):.1f}% of bytes removed)")
