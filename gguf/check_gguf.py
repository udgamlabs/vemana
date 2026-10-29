"""Check Vemana's GGUF files against the transformers model (Mac, $0, light).

1. Tokenization: llama.cpp (`llama-tokenize`) vs the released tokenizer on the first N te-2026-10 pages (with BOS).
2. Greedy continuations of four prompts, 40 tokens: transformers float32 vs llama.cpp (`llama-completion --temp 0`).
3. Perplexity the way `llama-perplexity` computes it (chunks of CTX tokens with BOS first, the second half of each
   chunk scored) on the same text, from transformers float32 and from each GGUF.

  python gguf/check_gguf.py MODEL_DIR GGUF [GGUF ...] --out RESULTS.json   # llama.cpp tools on PATH
"""
import argparse
import json
import math
import re
import subprocess
import tempfile

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PAGES = "eval/te-2026-10/data/test.parquet"   # built with eval/build_te.py
PROMPTS = ["వర్షాకాలంలో", "కంప్యూటర్ అనేది", "తెలుగు భాష", "ఒక రోజు ఒక రైతు"]


def llama_ids(gguf, text):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(text)
    out = subprocess.run(["llama-tokenize", "-m", gguf, "-f", f.name, "--ids", "--log-disable"],
                         capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1]
    return json.loads(out)


def llama_greedy(gguf, prompt, n):
    out = subprocess.run(["llama-completion", "-m", gguf, "-p", prompt, "-n", str(n), "--temp", "0", "--no-display-prompt",
                          "-c", "1024"], capture_output=True, text=True, check=True).stdout   # --log-disable would also hide the text
    return out[1:] if out.startswith(" ") else out   # llama.cpp keeps the space SentencePiece puts before the first piece


def llama_ppl(gguf, path, ctx, chunks):
    out = subprocess.run(["llama-perplexity", "-m", gguf, "-f", path, "-c", str(ctx), "--chunks", str(chunks), "-b", str(ctx)],
                         capture_output=True, text=True, check=True)
    m = re.search(r"Final estimate: PPL = ([0-9.]+)", out.stdout + out.stderr)
    return float(m.group(1)) if m else None


def hf_ppl(model, ids, ctx, chunks, bos):
    """llama-perplexity's scheme: chunk i = ids[i*ctx:(i+1)*ctx] with its first token replaced by BOS; score the
    tokens in the second half of the chunk."""
    nll, n = 0.0, 0
    with torch.no_grad():
        for i in range(chunks):
            c = list(ids[i * ctx:(i + 1) * ctx])
            if len(c) < ctx:
                break
            c[0] = bos
            x = torch.tensor([c])
            lp = torch.log_softmax(model(x).logits[0].float(), -1)
            first = ctx // 2
            for j in range(first, ctx - 1):
                nll -= lp[j, c[j + 1]].item()
                n += 1
    return math.exp(nll / n), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("ggufs", nargs="+")
    ap.add_argument("--pages", type=int, default=40)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--chunks", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    model = AutoModelForCausalLM.from_pretrained(a.model_dir, dtype=torch.float32).eval()
    texts = pq.read_table(PAGES, columns=["text"]).column("text").to_pylist()
    res = {"model_dir": a.model_dir, "ggufs": {}}
    # 1. tokenization
    for g in a.ggufs:
        bad = [i for i, t in enumerate(texts[:a.pages]) if llama_ids(g, t) != tok(t)["input_ids"]]
        res["ggufs"][g] = {"tokenization_pages_differing": len(bad), "pages_checked": a.pages, "first_bad": bad[:3]}
        print(g, "tokenization differing pages:", len(bad), "of", a.pages, flush=True)
    # 2. greedy
    res["greedy_hf"] = {}
    for p in PROMPTS:
        x = tok(p, return_tensors="pt")
        o = model.generate(**x, max_new_tokens=40, do_sample=False)
        res["greedy_hf"][p] = tok.decode(o[0][x["input_ids"].shape[1]:], skip_special_tokens=True)
    for g in a.ggufs:
        same = {}
        for p in PROMPTS:
            got = llama_greedy(g, p, 40)
            ref = res["greedy_hf"][p]
            k = 0
            while k < min(len(got), len(ref)) and got[k] == ref[k]:
                k += 1
            same[p] = {"common_prefix_chars": k, "ref_chars": len(ref), "gguf": got[:200]}
        res["ggufs"][g]["greedy"] = same
        print(g, "greedy common prefix chars:", {p: v["common_prefix_chars"] for p, v in same.items()}, flush=True)
    # 3. perplexity on the same text
    text = "\n\n".join(texts[:120])
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(text)
    ids = tok(text)["input_ids"]
    ppl, n = hf_ppl(model, ids, a.ctx, a.chunks, tok.bos_token_id)
    res["ppl_hf_fp32"] = {"ppl": ppl, "tokens_scored": n, "ctx": a.ctx, "chunks": a.chunks}
    print("transformers fp32 PPL", round(ppl, 4), "tokens", n, flush=True)
    for g in a.ggufs:
        p = llama_ppl(g, f.name, a.ctx, a.chunks)
        res["ggufs"][g]["ppl"] = p
        print(g, "llama.cpp PPL", p, flush=True)
    json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
