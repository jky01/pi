"""情節記憶（海馬迴）：外部 store + 檢索。寫入是一次性、加法的 → 零干擾、零遺忘、零侵蝕
（權重完全不動）。檢索用輕量 lexical overlap（這些事實的問句與陳述句共享稀有實體詞,overlap
檢索已近乎完美;production 可換成語意 embedder,介面不變）。同時記每條的出現次數,供門控用。"""
import re


def _toks(s):
    return set(re.findall(r"[a-z0-9]+", s.lower()))


class EpisodicMemory:
    def __init__(self):
        self.items = []  # 每條: dict(text, toks, count)

    def write(self, text):
        for it in self.items:
            if it["text"] == text:
                it["count"] += 1
                return it
        it = dict(text=text, toks=_toks(text), count=1)
        self.items.append(it)
        return it

    def retrieve(self, query, k=3):
        q = _toks(query)
        scored = sorted(self.items, key=lambda it: len(q & it["toks"]), reverse=True)
        return [it["text"] for it in scored[:k] if len(q & it["toks"]) > 0]

    def hot(self, min_count):
        """門控：出現次數 >= min_count 的條目（值得鞏固進權重的）。"""
        return [it["text"] for it in self.items if it["count"] >= min_count]

    def __len__(self):
        return len(self.items)
