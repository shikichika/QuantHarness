"""
QuantHarness を日本株（や任意の Yahoo Finance ティッカー）で1回動かし、判断をログに残すスクリプト。

使い方:
  export ANTHROPIC_API_KEY=...        # または OPENAI_API_KEY
  python run_jp.py 7203.T                     # トヨタ, 1時間足
  python run_jp.py 7203.T --interval 1d       # 日足
  python run_jp.py 6758.T --provider openai --model gpt-4o
  python run_jp.py --csv mydata.csv --name 7203.T   # 自前データ（Datetime,Open,High,Low,Close）

判断は decisions.jsonl に1行ずつ追記される。答え合わせは evaluate_jp.py で行う。

定期実行（GitHub Actions）向けのオプション:
  --drop-partial   まだ確定していない最新の足を捨てる
  --skip-stale     最新の足が今日のものでなければ（休場日）何もしない
  同じ銘柄・同じ最終足の判断がログにすでにあれば、LLM を呼ばずにスキップする。
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import static_util
from default_config import DEFAULT_CONFIG
from trading_graph import TradingGraph

BARS = 45  # 論文・Web UI と同じく直近45本を渡す
FORECAST_BARS = 3  # 何本先を予想させるか（論文と同じ）。evaluate_jp.py もこの本数で採点する
PERIOD_FOR_INTERVAL = {"1h": "60d", "4h": "60d", "1d": "6mo", "1wk": "5y"}
BAR_LENGTH = {"1h": pd.Timedelta(hours=1), "4h": pd.Timedelta(hours=4), "1d": pd.Timedelta(days=1), "1wk": pd.Timedelta(days=7)}
DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-5-5",
    "openai": "gpt-4o",
}


def load_yfinance(ticker: str, interval: str) -> pd.DataFrame:
    import yfinance as yf

    yf_interval = "1h" if interval == "4h" else interval
    for attempt in range(3):  # Yahoo はクラウドの IP から 429 を返すことがあるので数回試す
        df = yf.download(
            ticker,
            period=PERIOD_FOR_INTERVAL.get(interval, "60d"),
            interval=yf_interval,
            progress=False,
            auto_adjust=False,
        )
        if not df.empty:
            break
        time.sleep(10 * (attempt + 1))
    if df.empty:
        raise SystemExit(f"データが取れませんでした: {ticker} ({interval})")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close"]].dropna()
    if interval == "4h":
        # 東証は昼休みがあるので、4時間足は日中を前場/後場でまとめた近似になる
        df = df.resample("4h", origin="start_day").agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
        ).dropna()
    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: "Datetime"})
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    return df


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    return df[["Datetime", "Open", "High", "Low", "Close"]].dropna()


def to_kline_dict(df: pd.DataFrame) -> dict:
    s = df.tail(BARS).reset_index(drop=True)
    return {
        "Datetime": s["Datetime"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist(),
        "Open": s["Open"].astype(float).tolist(),
        "High": s["High"].astype(float).tolist(),
        "Low": s["Low"].astype(float).tolist(),
        "Close": s["Close"].astype(float).tolist(),
    }


def parse_decision(raw) -> dict:
    text = raw if isinstance(raw, str) else getattr(raw, "content", str(raw))
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    return {"raw": text}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ticker", nargs="?", help="例: 7203.T, 6758.T, ^N225, BTC-USD")
    p.add_argument("--interval", default="1h", choices=["1h", "4h", "1d", "1wk"])
    p.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"])
    p.add_argument("--model", default=None, help="画像入力に対応したモデルを指定すること")
    p.add_argument("--csv", default=None)
    p.add_argument("--name", default=None, help="--csv 使用時の銘柄名")
    p.add_argument("--log", default="decisions.jsonl")
    p.add_argument("--lang", default="Japanese", help="レポートの言語。英語のままにするなら English")
    p.add_argument("--drop-partial", action="store_true", help="確定していない最新の足を捨てる")
    p.add_argument("--skip-stale", action="store_true", help="最新の足が今日のものでなければ何もしない")
    args = p.parse_args()

    if args.csv:
        df = load_csv(args.csv)
        name = args.name or Path(args.csv).stem
    elif args.ticker:
        df = load_yfinance(args.ticker, args.interval)
        name = args.ticker
    else:
        p.error("ticker か --csv を指定してください")

    tz = df["Datetime"].dt.tz
    now = pd.Timestamp.now(tz=tz) if tz is not None else None
    if args.drop_partial and now is not None:
        df = df[df["Datetime"] + BAR_LENGTH[args.interval] <= now]
    if args.skip_stale and now is not None and not df.empty:
        if df["Datetime"].iloc[-1].date() != now.date():
            print(f"{name}: 最新の足が今日のものではないのでスキップ（{df['Datetime'].iloc[-1]}）")
            return

    if len(df) < BARS:
        raise SystemExit(f"データが足りません（{len(df)}本 < {BARS}本）")

    kline = to_kline_dict(df)

    if Path(args.log).exists():
        for line in open(args.log, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if (r.get("ticker"), r.get("interval"), r.get("last_bar")) == (name, args.interval, kline["Datetime"][-1]):
                print(f"{name}: 最終足 {r['last_bar']} の判断は記録済みなのでスキップ")
                return

    config = DEFAULT_CONFIG.copy()
    model = args.model or DEFAULT_MODELS[args.provider]
    config.update(
        agent_llm_provider=args.provider,
        graph_llm_provider=args.provider,
        agent_llm_model=model,
        graph_llm_model=model,
    )
    if args.provider == "anthropic":
        config["anthropic_api_key"] = ""  # 環境変数 ANTHROPIC_API_KEY を使う
        # 新しい Claude モデルは temperature のデフォルト以外の値を受け付けない
        config["agent_llm_temperature"] = None
        config["graph_llm_temperature"] = None
    else:
        config["api_key"] = ""  # 環境変数 OPENAI_API_KEY を使う

    graph = TradingGraph(config=config)
    state = {
        "kline_data": kline,
        "analysis_results": None,
        "messages": [],
        "time_frame": {"1h": "1hour", "4h": "4hour", "1d": "1day", "1wk": "1 week"}[args.interval],
        "stock_name": name,
        "output_language": args.lang,
        "forecast_bars": FORECAST_BARS,
        "pattern_image": static_util.generate_kline_image(kline)["pattern_image"],
        "trend_image": static_util.generate_trend_image(kline)["trend_image"],
    }

    final = graph.graph.invoke(state)
    decision = parse_decision(final.get("final_trade_decision"))

    record = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "ticker": name,
        "interval": args.interval,
        "provider": args.provider,
        "model": model,
        "last_bar": kline["Datetime"][-1],
        "last_close": kline["Close"][-1],
        "decision": decision,
        "reports": {
            "indicator": str(final.get("indicator_report", "")),
            "pattern": str(final.get("pattern_report", "")),
            "trend": str(final.get("trend_report", "")),
        },
    }
    with open(args.log, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(json.dumps(decision, ensure_ascii=False, indent=2))
    print(f"\n→ {args.log} に記録しました（最終足: {record['last_bar']}, 終値: {record['last_close']}）")


if __name__ == "__main__":
    main()
