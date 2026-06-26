"""
產生並快取 pi 的十進位小數位數字序列。

這個序列被用作持續學習 (continual learning) 實驗的資料來源：
- pi 的小數位在統計上近似均勻分布、彼此近乎獨立（沒有已知的短週期重複），
  因此可以當作一個「決定性、可重現，但近乎不重複」的無限資料串流，
  非常適合拿來跑長時間的線上持續學習診斷（呼應對話中 Stage C 的設計精神）。
"""
import os
import mpmath


def generate_pi_digits(n_digits: int, cache_path: str) -> str:
    """回傳 pi 小數點後 n_digits 位數字組成的字串（不含 '3.'），並快取到檔案。"""
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            cached = f.read().strip()
        if len(cached) >= n_digits:
            return cached[:n_digits]

    mpmath.mp.dps = n_digits + 15
    s = mpmath.nstr(mpmath.pi, n_digits + 10)
    # s 形如 '3.14159...'，取小數點後 n_digits 位
    int_part, frac_part = s.split(".")
    digits = frac_part[:n_digits]
    assert len(digits) == n_digits, f"only got {len(digits)} digits"

    with open(cache_path, "w") as f:
        f.write(digits)
    return digits


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    cache_path = os.path.join(out_dir, "pi_digits_600000.txt")
    digits = generate_pi_digits(600_000, cache_path)
    print(f"generated/cached {len(digits)} digits at {cache_path}")
    print("first 50:", digits[:50])
    # 簡單分布檢查：每個數字 0-9 出現頻率應接近 1/10
    from collections import Counter
    c = Counter(digits)
    print("digit frequency (first 100000):", {k: round(v / 100000, 4) for k, v in Counter(digits[:100000]).items()})
