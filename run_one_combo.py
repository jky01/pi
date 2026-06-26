"""
單次跑一個 (method, seed) 組合，把結果以一行 JSON 附加到 results_partial.jsonl。
拆成單次呼叫是因為 sandbox 的 bash 呼叫有逐次的時間上限，完整跑 4 method x 3 seed
（n_tasks=80）一次性執行容易超時；拆開後每個組合各自獨立、可重複執行、可累積進度。
"""
import json
import os
import sys
import time

from run import run_one, summarize

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PARTIAL_PATH = os.path.join(OUT_DIR, "results_partial.jsonl")


def already_done(method, seed, mode):
    if not os.path.exists(PARTIAL_PATH):
        return False
    with open(PARTIAL_PATH) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("method") == method and rec.get("seed") == seed and rec.get("mode") == mode:
                return True
    return False


def main():
    method = sys.argv[1]
    seed = int(sys.argv[2])
    mode = "label_permuted"
    if len(sys.argv) > 3:
        mode = sys.argv[3]

    if already_done(method, seed, mode):
        print(f"skip {method} seed={seed} mode={mode}：已經跑過了")
        return

    stream_kwargs = dict(
        digits_file="pi_digits_600000.txt",
        K=8, n_tasks=80, steps_per_task=4000, test_per_task=300,
        mode=mode,
    )
    model_kwargs = dict(in_dim=80, h1=64, h2=64, out_dim=10)
    lr = 0.1

    t0 = time.time()
    res = run_one(method, seed, dict(stream_kwargs), model_kwargs, lr=lr)
    summary = summarize(res)
    res["summary"] = summary
    dt = time.time() - t0

    with open(PARTIAL_PATH, "a") as f:
        f.write(json.dumps(res) + "\n")

    print(f"[{method} seed={seed} mode={mode}] final_avg_acc={summary['final_avg_acc']:.3f} "
          f"bwt={summary['bwt']:.3f} diag_first={summary['diag'][0]:.3f} "
          f"diag_last={summary['diag'][-1]:.3f}  ({dt:.1f}s)")


if __name__ == "__main__":
    main()
