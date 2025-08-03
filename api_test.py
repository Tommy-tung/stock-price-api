from flask import Flask, request, jsonify
import pandas as pd
import pandas as pd
import datetime
import time
import urllib.request
import json
import ssl
from dateutil.relativedelta import relativedelta
import numpy as np
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier.efficient_semivariance import EfficientSemivariance
from pypfopt.efficient_frontier.efficient_cvar import EfficientCVaR
# import matplotlib.pyplot as plt
import cvxpy as cp

# 下載股價資料
def pricedata(symbol):
    today = datetime.date.today()
    oneyear = today - relativedelta(years=3)
    timestamp = int(time.mktime(oneyear.timetuple()))
    timestamp2 = int(time.mktime(today.timetuple()))
    while True:
        try:
            url = f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}?period1={timestamp}&period2={timestamp2}&interval=1d&events=history&includeAdjustedClose=true"
            headers = {'User-Agent': 'Mozilla/5.0'}
            context = ssl._create_unverified_context()
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, context=context) as jsondata:
                j = json.loads(jsondata.read().decode('utf-8-sig'))
                timestamps = j['chart']['result'][0]['timestamp']
                closes = j['chart']['result'][0]['indicators']['quote'][0]['close']
                df = pd.DataFrame({'timestamp': timestamps, 'Close': closes})
                df['Date'] = pd.to_datetime(df['timestamp'], unit='s')
                df.set_index('Date', inplace=True)
                return df[['Close']].dropna()
        except Exception as e:
            print(f"錯誤: {e}")
            time.sleep(5)

# Logging setup
today_str = datetime.date.today().strftime("%Y%m%d")
log_filename = f"data_selection_log_{today_str}.txt"

def log(msg):
    with open(log_filename, "a", encoding="utf-8") as f:
        print(msg)
        f.write(msg + "\n")

def clean_symbol(code):
    return str(code).split()[0] if pd.notna(code) else ""

def needs_price_data(label):
    price_keywords = {"現價", "收盤價", "close", "Close"}
    # Detect if the query references price variables
    if label.get('query_type') == "price_query":
        return True
    return any(col in price_keywords for col in label.get('columns', []))

def select_symbols(df: pd.DataFrame, label_filters: list) -> list:
    try:
        overall_mask = pd.Series([True] * len(df))
        price_labels = []
        for label in label_filters:
            qtype = label.get('query_type')
            missing_cols = [col for col in label.get('columns', []) if col not in df.columns]
            if needs_price_data(label):
                price_labels.append(label)
                log(f"[INFO] Label '{label['label_zh']}' requires price data. Will use pricedata(symbol) for evaluation.")
                continue # Will handle after other filters
            elif missing_cols:
                log(f"[ERROR] Label '{label['label_zh']}' missing columns in data: {missing_cols}. Skipping this filter.")
                return []
            else:
                log(f"[INFO] Applying label '{label['label_zh']}' (type: {qtype})...")

            # --- Fix: ensure relevant columns are numeric (for numeric comparison queries)
            if qtype in ("condition", "condition_and", "condition_or", "percentile", "percentile_and"):
                for col in label.get('columns', []):
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

            # Non-price-data queries
            if qtype in ("condition", "condition_and", "condition_or"):
                mask = eval(label['query'], {"df": df})
            elif qtype == "percentile":
                col = label['columns'][0]
                p = label['percentile'] / 100.0
                mask = df[col] <= df[col].quantile(p) if p < 0.5 else df[col] >= df[col].quantile(p)
            elif qtype == "percentile_and":
                mask = pd.Series([True] * len(df))
                percentiles = label.get('percentiles', {})
                for col, perc in percentiles.items():
                    p = perc / 100.0
                    mask &= df[col] <= df[col].quantile(p) if p < 0.5 else df[col] >= df[col].quantile(p)
                if "query" in label and "&" in label["query"]:
                    parts = label["query"].split("&", 1)
                    extra_cond = parts[1].strip()
                    mask &= eval(extra_cond, {"df": df})
            elif qtype == "keyword":
                mask = pd.Series([False] * len(df))
                for col in label['columns']:
                    for kw in label['keywords']:
                        mask |= df[col].astype(str).str.contains(kw, case=False, na=False)
            else:
                log(f"[ERROR] Unknown query_type: {qtype} in label '{label['label_zh']}'.")
                return []

            overall_mask &= mask
            log(f"[INFO] After label '{label['label_zh']}', {overall_mask.sum()} stocks remain.")

        # First filter using non-price-data conditions
        filtered_df = df[overall_mask].copy()
        log(f"[INFO] {len(filtered_df)} symbols match all non-price-data conditions.")

        # If there are price-based labels, filter further
        final_symbols = []
        for idx, row in filtered_df.iterrows():
            symbol_raw = row['代碼']
            symbol = clean_symbol(symbol_raw)
            keep = True
            for label in price_labels:
                log(f"[INFO] Checking price-data label '{label['label_zh']}' for {symbol} ...")
                try:
                    price_df = pricedata(symbol)
                    # Prepare local variables for eval
                    latest_close = price_df['Close'].iloc[-1]
                    local_vars = {
                        "price_df": price_df,
                        "latest_close": latest_close,
                        "pd": pd,
                        "symbol": symbol
                    }
                    # Now eval label['query'] which should use latest_close, price_df, etc.
                    if not eval(label['query'], {}, local_vars):
                        keep = False
                        log(f"[INFO] Symbol {symbol} does not match price-data label '{label['label_zh']}' (Query: {label['query']}).")
                        break
                except Exception as e:
                    log(f"[ERROR] Failed to fetch or process price data for {symbol}: {e}")
                    keep = False
                    break
            if keep:
                final_symbols.append(symbol)

        # If no price labels, just return the symbol list
        if not price_labels:
            final_symbols = [clean_symbol(x) for x in filtered_df['代碼']]

        if not final_symbols:
            log("[ERROR] No symbols match all filter conditions or all failed price data query.")
        else:
            log(f"[INFO] Final selected pool: {final_symbols}")
        return final_symbols
    except Exception as ex:
        log(f"[EXCEPTION] {ex}")
        return []

# Example main block



# log(f"[RESULT] pool = {pool}")


# 下載股價資料
def pricedata(symbol):
    today = datetime.date.today()
    oneyear = today - relativedelta(years=3)
    timestamp = int(time.mktime(oneyear.timetuple()))
    timestamp2 = int(time.mktime(today.timetuple()))
    while True:
        try:
            url = f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}?period1={timestamp}&period2={timestamp2}&interval=1d&events=history&includeAdjustedClose=true"
            headers = {'User-Agent': 'Mozilla/5.0'}
            context = ssl._create_unverified_context()
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, context=context) as jsondata:
                j = json.loads(jsondata.read().decode('utf-8-sig'))
                timestamps = j['chart']['result'][0]['timestamp']
                closes = j['chart']['result'][0]['indicators']['quote'][0]['close']
                df = pd.DataFrame({'timestamp': timestamps, 'Close': closes})
                df['Date'] = pd.to_datetime(df['timestamp'], unit='s')
                df.set_index('Date', inplace=True)
                return df[['Close']].dropna()
        except Exception as e:
            print(f"錯誤: {e}")
            time.sleep(5)

# Risk Parity 計算函數（使用 cvxpy）
def compute_risk_parity_weights(cov_matrix):
    n = cov_matrix.shape[0]
    w = cp.Variable(n)
    objective = cp.Minimize(cp.sum(-cp.log(w)) + cp.quad_form(w, cov_matrix.values))
    constraints = [cp.sum(w) == 1, w >= 1e-6] # 加上 lower bound 防止 log(0)
    problem = cp.Problem(objective, constraints)
    problem.solve()
    return pd.Series(w.value, index=cov_matrix.columns)


# 股票清單與 S&P 500
# pool = ['AAPL','MSFT','MCD','META','AMZN']
def portfolio(tickerslist):
    tickers = tickerslist
    benchmark = '^GSPC'

    price_dict = {}
    for ticker in tickers:
        df = pricedata(ticker)
        if df is not None:
            price_dict[ticker] = df['Close']
    price_df = pd.concat(price_dict.values(), axis=1)
    price_df.columns = tickers
    price_df.dropna(inplace=True)

    # 抓 S&P 500 並對齊時間
    sp500 = pricedata(benchmark)
    sp500 = sp500.reindex(price_df.index).dropna()
    price_df = price_df.loc[sp500.index]

    # 計算報酬與風險
    returns = price_df.pct_change().dropna()
    mu = mean_historical_return(price_df)
    S = CovarianceShrinkage(price_df).ledoit_wolf()

    # 儲存所有模型配置
    portfolios = {}

    # Max Sharpe
    ef_sharpe = EfficientFrontier(mu, S)
    ef_sharpe.max_sharpe()
    portfolios["Max Sharpe"] = ef_sharpe.clean_weights()

    # Min Volatility
    ef_vol = EfficientFrontier(mu, S)
    ef_vol.min_volatility()
    portfolios["Min Volatility"] = ef_vol.clean_weights()

    # Min Semivariance
    es = EfficientSemivariance(mu, price_df)
    es.min_semivariance()
    portfolios["Min Semivariance"] = es.clean_weights()

    # Min CVaR
    ec = EfficientCVaR(mu, price_df)
    ec.min_cvar()
    portfolios["Min CVaR"] = ec.clean_weights()

    # Risk Parity (用 cvxpy)
    rp_weights = compute_risk_parity_weights(S)
    portfolios["Risk Parity"] = rp_weights.to_dict()

    # 回測比較
    results = []
    cumulative_returns = {}
    weights_output = {}

    for model_name, weights in portfolios.items():
        w = pd.Series(weights).reindex(tickers).fillna(0)
        weights_output[model_name] = w
        port_ret = (returns * w).sum(axis=1)
        cumulative = (1 + port_ret).cumprod()
        cumulative_returns[model_name] = cumulative
        total_ret = cumulative.iloc[-1] - 1
        annual_ret = port_ret.mean() * 252
        volatility = port_ret.std() * np.sqrt(252)
        sharpe = annual_ret / volatility if volatility != 0 else 0
        drawdown = (cumulative / cumulative.cummax() - 1).min()
        results.append({
            "Model": model_name,
            "Total Return": total_ret,
            "Annual Return": annual_ret,
            "Volatility": volatility,
            "Sharpe Ratio": sharpe,
            "Max Drawdown": drawdown
        })

    # 加入 S&P 500 比較
    sp500_ret = sp500['Close'].pct_change().dropna()
    sp500_cum = (1 + sp500_ret).cumprod()
    cumulative_returns["S&P 500"] = sp500_cum
    sp500_ret = sp500_ret.loc[returns.index]

    total_ret = sp500_cum.iloc[-1] - 1
    annual_ret = sp500_ret.mean() * 252
    volatility = sp500_ret.std() * np.sqrt(252)
    sharpe = annual_ret / volatility if volatility != 0 else 0
    drawdown = (sp500_cum / sp500_cum.cummax() - 1).min()

    results.append({
        "Model": "S&P 500",
        "Total Return": total_ret,
        "Annual Return": annual_ret,
        "Volatility": volatility,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": drawdown
    })

    # 輸出績效表格
    df_result = pd.DataFrame(results)
    df_weights = pd.DataFrame(weights_output).fillna(0)

    return df_result, df_weights
    



app = Flask(__name__)

@app.route('/', methods=['POST'])
def echo():
    # print(request)
    data = request.get_json()
    df = pd.read_excel('s&p500_data.xlsx')
    label_filters = [
    {
        "label_zh": "低波動",
        "label_en": "Low Volatility",
        "description": "60日波動率在所有股票下30%，且Beta < 1",
        "columns": ["60日波動率", "Beta"],
        "query_type": "percentile_and",
        "percentiles": {"60日波動率": 30},
        "query": "(df['60日波動率'] <= df['60日波動率'].quantile(0.3)) & (df['Beta'] < 1)"
    },
    {
        "label_zh": "價值股",
        "label_en": "Value Stock",
        "description": "市值在所有股票中排名後30%",
        "columns": ["市值"],
        "query_type": "percentile",
        "percentile": 30,
        "query": "df['市值'] <= df['市值'].quantile(0.3)"
    }
]
    pool = select_symbols(df, label_filters)
    df_result, df_weights = portfolio(pool)
    # stock_list = df['代碼'].values[:5].tolist()
    # print("Received data:", data)


    return jsonify({"result": df_result, "weight" : df_weights})

if __name__ == '__main__':
    app.run()