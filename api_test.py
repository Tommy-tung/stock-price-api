from flask import Flask, request, jsonify, render_template_string, url_for
import pandas as pd
import datetime
import time
import json
import ssl
import re
from dateutil.relativedelta import relativedelta
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier.efficient_semivariance import EfficientSemivariance
from pypfopt.efficient_frontier.efficient_cvar import EfficientCVaR
from pypfopt.risk_models import sample_cov
# import matplotlib.pyplot as plt
import cvxpy as cp
import requests
import numpy as np
from html_format import generate_investment_report_html
import uuid
import os 

# 下載股價資料
def clean_query(query_str):
    return query_str.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")



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
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()  # 如果 HTTP 錯誤會丟出例外
            
            j = response.json()
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
    problem.solve(solver=cp.SCS)
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
    print(price_df)

    # 計算報酬與風險
    returns = price_df.pct_change().dropna()
    mu = mean_historical_return(price_df)
    print(mu, type(mu))
    S = CovarianceShrinkage(price_df).ledoit_wolf()
    # S = S.values.astype(np.float64)
    print(S, type(S))

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
    print(data)
    name = data.get('name')
    json_string = data.get('stock_json')
    df = pd.read_excel('s&p500_data.xlsx')
    # df = pd.read_excel('/Users/tommy84729/富邦/黑客松/stock_price_api/s&p500_data.xlsx')
    # json_string = """{
    #     "investing_labels": [
    #         {
    #         "label_zh": "穩健波動",
    #         "label_en": "Low Volatility",
    #         "description": "60日波動率在所有股票下30%，且 Beta < 1",
    #         "columns": [{"item": "60日波動率"}, {"item": "Beta"}],
    #         "query_type": "percentile_and",
    #         "percentiles": {"60日波動率": 30},
    #         "query": "(df['60日波動率'] <= df['60日波動率'].quantile(0.3)) & (df['Beta'] < 1)"
    #         }
    #     ],
    #     "industry_labels": [
    #         {
    #         "label_zh": "科技股",
    #         "label_en": "Technology Stock",
    #         "description": "產業分類相關欄位包含『科技』、『半導體』、『資訊』等關鍵字",
    #         "columns": [{"item": "GICS行業板塊"}, {"item": "Class L4 Nm"}, {"item": "當地分類簡介"}],
    #         "query_type": "keyword",
    #         "keywords": [{"item": "科技"}, {"item": "半導體"}, {"item": "資訊"}],
    #         "query": "欄位中包含任一關鍵字"
    #         }
    #     ]
    # }"""
    # json_str = json.dumps(json_string)       # 轉為 JSON 字串
    data = json.loads(json_string)
    combined_mask = pd.Series([True] * len(df))

    # 遍歷所有可能是 label 的欄位（只處理 list 結構）
    for label_group in data.values():
        if isinstance(label_group, list):
            for label in label_group:
                query_type = label.get("query_type", "")
                query = label.get("query", "")
                label_name = label.get("label_zh", "未命名標籤")

                label_mask = pd.Series([True] * len(df))  # 預設為全 True

                if query_type == "keyword" and query == "欄位中包含任一關鍵字":
                    columns = [col["item"] for col in label.get("columns", [])]
                    keywords = [kw["item"] for kw in label.get("keywords", [])]

                    label_mask = df[columns].apply(
                        lambda col: col.astype(str).apply(lambda val: any(keyword in val for keyword in keywords))
                    ).any(axis=1)

                else:
                    try:
                        cleaned_query = clean_query(query)
                        label_mask = eval(cleaned_query, {"df": df, "pd": pd})
                    except Exception as e:
                        print(f"⚠️ 無法執行「{label_name}」的 query：{e}")
                        continue

                # 將目前條件與累積條件取交集
                combined_mask &= label_mask
                print(f"✅ 套用「{label_name}」條件後剩下的資料列：")
                print(df[label_mask])

# 套用所有條件後的結果
    final_result = df[combined_mask]
    pool=final_result['代碼'].tolist()
    pool = [item.split()[0] for item in pool]
    df_result, df_weights = portfolio(pool)
    # stock_list = df['代碼'].values[:5].tolist()
    # print("Received data:", data)
    df_result = df_result[['Model','Annual Return', 'Max Drawdown',  'Sharpe Ratio','Total Return', 'Volatility']]
 
    df_intro = pd.DataFrame(['a', 'b'])
    df_weights_with_index = df_weights.copy()
    df_weights_with_index.insert(0, "個股標的", pool)

    html_result = generate_investment_report_html(df_result, df_weights_with_index, df_intro)

    # 產生唯一報表名稱
    report_id = str(uuid.uuid4())
    filename = f"{report_id}.html"

    # 使用 Flask root_path 建立儲存目錄
    reports_dir = os.path.join(app.root_path, "static", "reports")
    os.makedirs(reports_dir, exist_ok=True)

    # 寫入 HTML 檔案
    html_path = os.path.join(reports_dir, filename)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_result)

    # 組出公開的網址（Render 會自動公開 static 資源）
    base_url = request.host_url.rstrip("/")
    public_url = f"{base_url}/static/reports/{filename}"

    return jsonify({
                    "message": "✅ 報表產生成功",
                    "url": public_url
                })


    # return jsonify({
    # "result": df_result.to_dict(orient="records"),
    # "weight": df_weights.to_dict(orient="records")
    # })
    # return render_template_string(html_result)

if __name__ == '__main__':
    app.run()