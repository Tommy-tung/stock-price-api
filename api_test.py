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
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import cvxpy as cp
import requests
import numpy as np
from html_format import generate_investment_report_html
import uuid
import psutil, os
import math

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


def risk_parity_inverse_vol(cov: pd.DataFrame) -> pd.Series:
    iv = 1 / np.sqrt(np.diag(cov.values))
    w = iv / iv.sum()
    return pd.Series(w, index=cov.index)

def get_model_weights(model_name, mu, S, price_df):
    name = str(model_name).lower()
    if name in ["max sharpe", "max_sharpe", "maxsharpe"]:
        ef = EfficientFrontier(mu, S); ef.max_sharpe()
        return pd.Series(ef.clean_weights())
    elif name in ["min volatility", "min_volatility", "minvol"]:
        ef = EfficientFrontier(mu, S); ef.min_volatility()
        return pd.Series(ef.clean_weights())
    elif name in ["min semivariance", "min_semivariance", "minsemi"]:
        es = EfficientSemivariance(mu, price_df); es.min_semivariance()
        return pd.Series(es.clean_weights())
    elif name in ["min cvar", "min_cvar", "cvar"]:
        ec = EfficientCVaR(mu, price_df); ec.min_cvar()
        return pd.Series(ec.clean_weights())
    elif name in ["risk parity", "risk_parity", "rp"]:
        return risk_parity_inverse_vol(S)
    else:
        raise ValueError("未知模型：%s" % model_name)

def _perf_stats(port_ret: pd.Series, ann_factor: int = 252):
    cum = (1 + port_ret).cumprod()
    total_ret = float(cum.iloc[-1] - 1)
    ann_ret = float(port_ret.mean() * ann_factor)
    vol = float(port_ret.std() * np.sqrt(ann_factor))
    sharpe = float(ann_ret / vol) if vol != 0 else 0.0
    mdd = float((cum / cum.cummax() - 1).min())
    return {"Total Return": total_ret, "Annual Return": ann_ret,
            "Volatility": vol, "Sharpe Ratio": sharpe,
            "Max Drawdown": mdd, "Cumulative": cum}

def _drawdown_stats(cum_curve: pd.Series):
    dd = cum_curve / cum_curve.cummax() - 1.0
    avg_dd = float(dd[dd < 0].mean()) if (dd < 0).any() else 0.0
    in_dd = dd < 0
    lengths, count = [], 0
    for flag in in_dd:
        if flag: count += 1
        elif count > 0:
            lengths.append(count); count = 0
    if count > 0: lengths.append(count)
    avg_len = float(np.mean(lengths)) if lengths else 0.0
    return avg_dd, avg_len

def _safe(x, eps=1e-12): return float(x) if np.isfinite(x) else eps
def _pos(x, eps=1e-12): return max(_safe(x), eps)
def _gmean(xs): return float(math.exp(sum(math.log(max(v,1e-12)) for v in xs)/len(xs)))

# === 核心：GeneratedAssets 風格比較表（無 Diversification）===
def build_ga_comparison_table(model_name, w, port_ret, price_df, sp500_ret):
    stats_m = _perf_stats(port_ret)
    stats_b = _perf_stats(sp500_ret)
    avg_dd_m, avg_len_m = _drawdown_stats(stats_m["Cumulative"])
    avg_dd_b, avg_len_b = _drawdown_stats(stats_b["Cumulative"])

    # Returns score
    returns_x = _gmean([
        (1 + _safe(stats_m["Total Return"])) / (1 + _safe(stats_b["Total Return"])),
        _pos(stats_m["Annual Return"]) / _pos(stats_b["Annual Return"]),
    ])

    # Stability score
    sharpe_ratio = _pos(stats_m["Sharpe Ratio"]) / _pos(stats_b["Sharpe Ratio"])
    mdd_ratio    = _pos(abs(stats_b["Max Drawdown"])) / _pos(abs(stats_m["Max Drawdown"]))
    avgdd_ratio  = _pos(abs(avg_dd_b)) / _pos(abs(avg_dd_m))
    len_ratio    = _pos(avg_len_b) / _pos(avg_len_m)
    stability_x  = _gmean([sharpe_ratio, mdd_ratio, avgdd_ratio, len_ratio])

    # Asset score = Returns × Stability 的幾何平均
    asset_score_x = _gmean([returns_x, stability_x])

    # 表格輸出
    data = [
        ["Asset score", "—", f"{asset_score_x:.2f}x", "—"],

        ["Returns", "Score", f"{returns_x:.2f}x", "1.00x"],
        ["Returns", "Total return",
         f"{stats_m['Total Return']*100:.2f}%", f"{stats_b['Total Return']*100:.2f}%"],
        ["Returns", "Annual return",
         f"{stats_m['Annual Return']*100:.2f}%", f"{stats_b['Annual Return']*100:.2f}%"],

        ["Stability", "Score", f"{stability_x:.2f}x", "1.00x"],
        ["Stability", "Sharpe ratio",
         f"{stats_m['Sharpe Ratio']:.2f}", f"{stats_b['Sharpe Ratio']:.2f}"],
        ["Stability", "Max drawdown",
         f"{stats_m['Max Drawdown']*100:.2f}%", f"{stats_b['Max Drawdown']*100:.2f}%"],
        ["Stability", "Average drawdown",
         f"{avg_dd_m*100:.2f}%", f"{avg_dd_b*100:.2f}%"],
        ["Stability", "Average DD duration (days)",
         f"{avg_len_m:.1f}", f"{avg_len_b:.1f}"],
    ]
    return pd.DataFrame(data, columns=["Category","Metric",model_name,"S&P 500"])

# === 根據屬性選模型 ===
def select_model_by_profile(profile, mu, S, price_df, returns):
    rp = profile.lower()
    if rp in ["保守","conservative"]:
        candidates = ["Min CVaR","Min Semivariance","Min Volatility"]
        best_name,best_w,best_sharpe=None,None,-np.inf
        for name in candidates:
            w = get_model_weights(name, mu, S, price_df).reindex(price_df.columns).fillna(0)
            sharpe = _perf_stats((returns*w).sum(axis=1))["Sharpe Ratio"]
            if sharpe > best_sharpe: best_name,best_w,best_sharpe=name,w,sharpe
        return best_name,best_w
    if rp in ["中性","neutral"]:
        name="Risk Parity"
        return name,get_model_weights(name, mu, S, price_df).reindex(price_df.columns).fillna(0)
    name="Max Sharpe"
    return name,get_model_weights(name, mu, S, price_df).reindex(price_df.columns).fillna(0)

# === 主流程 ===
def portfolio_by_profile(tickerslist, risk_profile, save_path):
    tickers = tickerslist
    benchmark='^GSPC'

    # 價格拼接
    price_dict={t:pricedata(t)['Close'] for t in tickers if pricedata(t) is not None}
    price_df=pd.concat(price_dict.values(),axis=1); price_df.columns=list(price_dict.keys()); price_df.dropna(inplace=True)
    sp500=pricedata(benchmark); sp500=sp500.reindex(price_df.index).dropna()
    price_df=price_df.loc[sp500.index]

    # 報酬
    returns=price_df.pct_change().dropna()
    mu=mean_historical_return(price_df)
    S=CovarianceShrinkage(price_df).ledoit_wolf(); S.index=price_df.columns; S.columns=price_df.columns

    # 選模型
    chosen_model,w=select_model_by_profile(risk_profile, mu, S, price_df, returns)

    # 投組績效 & 基準
    port_ret=(returns*w).sum(axis=1)
    stats=_perf_stats(port_ret)
    sp_ret=sp500['Close'].pct_change().dropna().loc[port_ret.index]
    sp_stats=_perf_stats(sp_ret)

    # === (1) 指定客戶模型 vs S&P500 績效比較表 ===
    results=pd.DataFrame([
        {"Model":chosen_model,**{k:v for k,v in stats.items() if k!="Cumulative"}},
        {"Model":"S&P 500",**{k:v for k,v in sp_stats.items() if k!="Cumulative"}}
    ])

    # === (2) 最終資產配置（%） === w

    # === (3) GeneratedAssets 風格比較表 ===
    ga_table=build_ga_comparison_table(chosen_model,w,port_ret,price_df,sp_ret)

    # 圖：累積報酬
    cum_df=pd.DataFrame({chosen_model:stats["Cumulative"],"S&P 500":sp_stats["Cumulative"]})
    plt.figure(figsize=(10,6))
    for col in cum_df.columns: 
        plt.plot(cum_df.index,cum_df[col],label=col)
    plt.title(f"Cumulative Return: {chosen_model} (Profile: {risk_profile}) vs S&P500")
    plt.legend()
    plt.grid(True)
    filename = f"{uuid.uuid4()}.png"
    path = os.path.join(save_path, filename)
    plt.savefig(path)
    plt.close()

    return {"profile":risk_profile,"chosen_model":chosen_model,"weights":w,
            "results":results,"ga_table":ga_table,"cumulative":filename}
    



app = Flask(__name__)

@app.route('/', methods=['POST'])
def echo():
    # print(request)
    data = request.get_json()
    print(data)
    name = data.get('name')
    pool = data.get('stock_list')
    risk_profile = data.get('risk_profile')
    # df = pd.read_excel('s&p500_data.xlsx')
    df = pd.read_excel('data2.xlsx')
    # df['代碼'] = [item.split()[0] for item in df['代碼']]
    # 使用 Flask root_path 建立儲存目錄
    reports_dir = os.path.join(app.root_path, "static", "reports")
    os.makedirs(reports_dir, exist_ok=True)


# 套用所有條件後的結果
    df_result = portfolio_by_profile(pool, risk_profile, reports_dir)

    portfolio_plot_filename = df_result['cumulative']
    df_weights_with_index = pd.DataFrame(df_result['weights'], columns = ['投資比例'])
    df_weights_with_index.insert(0, "個股標的", pool)
    df_weights_with_index = df_weights_with_index[df_weights_with_index['投資比例'] > 0]

    df_intro = df[df['代碼'].isin(df_weights_with_index['個股標的'].tolist())][['TR.TRBCBusinessSector' ,'簡介']]
    df_intro.insert(0, "個股標的", df_weights_with_index['個股標的'].tolist())
    plot_url = url_for('static', filename=f'reports/{portfolio_plot_filename}')
    
    html_result = generate_investment_report_html(df_weights_with_index[['個股標的', '投資比例']], df_result['ga_table'], df_intro, plot_url)

    # 產生唯一報表名稱
    report_id = str(uuid.uuid4())
    filename = f"{report_id}.html"

    # 寫入 HTML 檔案
    html_path = os.path.join(reports_dir, filename)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_result)

    # 組出公開的網址（Render 會自動公開 static 資源）
    base_url = request.host_url.rstrip("/")
    public_url = f"{base_url}/static/reports/{filename}"

    return public_url
    # return jsonify({
    #                 "message": "✅ 報表產生成功",
    #                 "url": public_url
    #             })


    # return jsonify({
    # "result": df_result.to_dict(orient="records"),
    # "weight": df_weights.to_dict(orient="records")
    # })
    # return render_template_string(html_result)

if __name__ == '__main__':
    app.run()