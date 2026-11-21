"""
run_jp.py が残した decisions.jsonl の答え合わせをする。

論文と同じ考え方で、判断した足の「次の3本」の終値が予測方向に動いたかを数える（0〜3点）。
あわせて、3本後の終値までのリターン（ショートは符号反転）も出す。

使い方:
  python evaluate_jp.py                 # decisions.jsonl を評価
  python evaluate_jp.py --log other.jsonl
  python evaluate_jp.py --out evaluation.csv   # 結果を CSV にも保存

取引時間中に実行すると作りかけの足と比べてしまうので、引け後に実行すること。
"""

import argparse
import json

import pandas as pd

from run_jp import load_yfinance

HORIZON = 3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", default="decisions.jsonl")
    p.add_argument("--out", default=None, help="結果を保存する CSV のパス")
    args = p.parse_args()

    try:
        records = [json.loads(l) for l in open(args.log, encoding="utf-8") if l.strip()]
    except FileNotFoundError:
        raise SystemExit(f"{args.log} がありません。先に run_jp.py で判断を記録してください"
                         f"（例: python run_jp.py 6758.T --log {args.log}）")
    cache, rows = {}, []
    for r in records:
        d = r["decision"].get("decision", "").upper()
        if d not in ("LONG", "SHORT"):
            continue
        key = (r["ticker"], r["interval"])
        if key not in cache:
            try:
                cache[key] = load_yfinance(*key).set_index("Datetime")["Close"]
            except SystemExit as e:  # 1銘柄取れなくても他は評価する
                print(e)
                cache[key] = None
        close = cache[key]
        if close is None:
            rows.append({**_base(r, d), "status": "取得失敗"})
            continue
        last = pd.Timestamp(r["last_bar"])
        if close.index.tz is not None and last.tz is None:
            last = last.tz_localize(close.index.tz)
        if close.index[0] > last:
            # yfinance の取得期間（1h足なら直近60日）より古い。評価すると無関係な足と比べてしまう
            rows.append({**_base(r, d), "status": "データ範囲外"})
            continue
        future = close[close.index > last].head(HORIZON)
        if len(future) < HORIZON:
            rows.append({**_base(r, d), "status": "未確定"})
            continue
        sign = 1 if d == "LONG" else -1
        prev = [r["last_close"]] + future.tolist()[:-1]
        moves = [b - a for a, b in zip(prev, future)]
        hits = sum(1 for m in moves if sign * m > 0)
        ret = sign * (future.iloc[-1] / r["last_close"] - 1)
        rows.append({
            **_base(r, d), "status": "確定", "hits(0-3)": hits, "return_%": round(ret * 100, 3),
            # ベースライン用（同じ足で「毎回LONG」「毎回SHORT」だったら何本当たったか）
            "_long_hits": sum(1 for m in moves if m > 0),
            "_short_hits": sum(1 for m in moves if m < 0),
            "_flat": sum(1 for m in moves if m == 0),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("評価できる記録がありません")
        return
    for c in ("hits(0-3)", "_long_hits", "_short_hits", "_flat"):
        if c in df:
            df[c] = df[c].astype("Int64")  # 未確定の行があっても 2.0 ではなく 2 と表示する
    pd.set_option("display.width", 160)
    print(df[[c for c in df.columns if not c.startswith("_")]].to_string(index=False))
    if args.out:
        df.to_csv(args.out, index=False, encoding="utf-8")
        print(f"\n→ {args.out} に保存しました")
    done = df[df["status"] == "確定"]
    if done.empty:
        return

    n = len(done) * HORIZON
    acc = done["hits(0-3)"].sum() / n
    long_acc = done["_long_hits"].sum() / n
    short_acc = done["_short_hits"].sum() / n
    rand_acc = (long_acc + short_acc) / 2
    flat = done["_flat"].sum() / n
    long_share = (done["decision"] == "LONG").mean()

    print(f"\n確定 {len(done)} 件（答え合わせ {n} 本）")
    print(f"  QuantHarness の正解率 : {acc:.1%}")
    print(f"  毎回LONGの正解率      : {long_acc:.1%}")
    print(f"  毎回SHORTの正解率     : {short_acc:.1%}")
    print(f"  ランダムの期待値      : {rand_acc:.1%}（変わらずの足 {flat:.1%} は常に外れ扱い）")
    print(f"  ランダムとの差        : {(acc - rand_acc) * 100:+.1f} ポイント"
          f" / 毎回LONGとの差: {(acc - long_acc) * 100:+.1f} ポイント")
    print(f"  LONG判断の割合        : {long_share:.0%}")
    print(f"  平均リターン          : {done['return_%'].mean():.3f}% / 勝率 {(done['return_%'] > 0).mean():.0%}")
    if len(done) < 30:
        print("  ※ 30件未満なので、まだ誤差の範囲です")

    if done["ticker"].nunique() > 1:
        by = done.groupby("ticker").agg(
            件数=("hits(0-3)", "size"),
            正解率=("hits(0-3)", lambda s: s.sum() / (len(s) * HORIZON)),
            毎回LONG=("_long_hits", lambda s: s.sum() / (len(s) * HORIZON)),
            平均リターン_pct=("return_%", "mean"),
        ).round(3)
        print("\n銘柄別:\n" + by.to_string())


def _base(r, d):
    return {"ticker": r["ticker"], "interval": r["interval"], "last_bar": r["last_bar"], "decision": d}


if __name__ == "__main__":
    main()
