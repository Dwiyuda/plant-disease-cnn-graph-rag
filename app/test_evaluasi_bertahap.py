# -*- coding: utf-8 -*-
"""
Tes kecil (bukan pytest, cukup `python test_evaluasi_bertahap.py`) untuk
memastikan evaluasi_bertahap() di evaluate_ragas.py benar-benar bisa
resume: item yang gagal (mis. rate limit) TIDAK ikut menghapus skor item
lain, dan run berikutnya hanya mengevaluasi ulang item yang gagal itu saja.

Tidak memanggil Groq/Neo4j sama sekali -- fungsi evaluate() RAGAS
dimonkeypatch dengan hasil palsu.
"""
import os
import json
import pandas as pd
import evaluate_ragas as ev

CACHE_PATH = "_test_scores_cache.json"


class _FakeMetric:
    def __init__(self, name):
        self.name = name


class _FakeResult:
    def __init__(self, row):
        self._row = row

    def to_pandas(self):
        return pd.DataFrame([self._row])


def main():
    if os.path.exists(CACHE_PATH):
        os.remove(CACHE_PATH)

    metrics = [_FakeMetric("faithfulness"), _FakeMetric("answer_relevancy")]
    records = [
        {"user_input": "q1", "retrieved_contexts": [], "response": "a1", "reference": "r1"},
        {"user_input": "q2", "retrieved_contexts": [], "response": "a2", "reference": "r2"},
        {"user_input": "q3", "retrieved_contexts": [], "response": "a3", "reference": "r3"},
    ]
    calls = {"n": 0}

    def evaluate_gagal_di_q2(dataset, metrics, llm, embeddings, run_config):
        calls["n"] += 1
        if dataset[0]["user_input"] == "q2":
            raise RuntimeError("simulasi rate limit 429")
        return _FakeResult({"faithfulness": 1.0, "answer_relevancy": 0.9})

    ev.evaluate = evaluate_gagal_di_q2
    ev.evaluasi_bertahap(records, metrics, None, None, None, cache_path=CACHE_PATH)
    assert calls["n"] == 3
    with open(CACHE_PATH, encoding="utf-8") as f:
        cache1 = json.load(f)
    key_q1, key_q2, key_q3 = (ev._cache_key(r) for r in records)
    assert "faithfulness" in cache1[key_q1] and "faithfulness" in cache1[key_q3]
    assert "faithfulness" not in cache1.get(key_q2, {})

    calls["n"] = 0

    def evaluate_selalu_ok(dataset, metrics, llm, embeddings, run_config):
        calls["n"] += 1
        return _FakeResult({"faithfulness": 1.0, "answer_relevancy": 0.9})

    # dataset diacak + disisipi item baru: skor lama tetap harus terpakai
    # lewat cache_key berbasis isi pertanyaan, bukan posisi index.
    records_diacak = [records[2], records[0],
                       {"user_input": "q4-baru", "retrieved_contexts": [], "response": "a4", "reference": "r4"},
                       records[1]]
    ev.evaluate = evaluate_selalu_ok
    df = ev.evaluasi_bertahap(records_diacak, metrics, None, None, None, cache_path=CACHE_PATH)
    assert calls["n"] == 2, "hanya q4-baru & q2 (yang gagal) semestinya dievaluasi ulang"
    assert len(df) == 4 and df["faithfulness"].notna().all()

    os.remove(CACHE_PATH)
    print("OK: evaluasi_bertahap resume dengan benar walau dataset diacak/ditambah item baru.")


if __name__ == "__main__":
    main()
