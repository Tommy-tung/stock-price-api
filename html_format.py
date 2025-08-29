from jinja2 import Template
import pandas as pd
from html import escape

def parse_score_to_pct(score_str: str) -> float:
    if not isinstance(score_str, str):
        return 0
    s = score_str.strip().lower().replace("×", "x")
    if s in {"—", "-", ""}:
        return 0
    try:
        if s.endswith("x"):
            v = float(s[:-1])
            return max(0, min(100, (v / 1) * 100.0))
        v = float(s)
        return max(0, min(100, (v / 1) * 100.0))
    except Exception:
        return 0

def get_category_score(group: pd.DataFrame) -> str:
    m = group[group["Metric"].isin(["Score", "—"])]
    if not m.empty:
        return str(m.iloc[0]["investment strategy"])
    return None


def generate_investment_report_html(name, intro_text, df_stocks, df_backtest, df_intro, plot_url):
    # 轉成 HTML 表格

    # df_intro_style = df_intro.style.set_td_classes({'簡介': 'text-start'})
    stocks_table_html = df_stocks.to_html(index=False, classes="table table-dark table-striped")
    # intro_table_html = df_intro_style.to_html(index=False, classes="table table-dark table-striped")
    td_classes = pd.DataFrame("", index=df_intro.index, columns=df_intro.columns)  # 先建一個空白 class 表
    td_classes["簡介"] = "text-start"  # 只有「簡介」這欄加上 class

    intro_table_html = (
        df_intro.style
            .hide(axis="index")
            .set_td_classes(td_classes)
            .set_table_styles([
                {'selector': 'th:nth-child(1), td:nth-child(1)', 'props': [('width', '15%')]},
                {'selector': 'th:nth-child(2), td:nth-child(2)', 'props': [('width', '10%')]},
                {'selector': 'th:nth-child(3), td:nth-child(3)', 'props': [('width', '15%')]},
                {'selector': 'th:nth-child(4), td:nth-child(4)', 'props': [('width', '60%'), ('text-align', 'left')]},
            ])
            .to_html(classes="table table-dark table-striped")
    )

    # 保持類別順序
    categories, seen = [], set()
    for c in df_backtest["Category"].tolist():
        if c not in seen:
            seen.add(c)
            categories.append(c)

    sections_html = []
    for cat in categories:
        g = df_backtest[df_backtest["Category"] == cat]
        score_str, pct = None, None
        score_str = get_category_score(g)
        sp500_score = g[g["Metric"].isin(["Score", "—"])]["S&P 500"]

        if cat == "Asset score":
            
            pct = parse_score_to_pct(score_str) if score_str else 0
            sp500_score_str = str(sp500_score.iloc[0]) if not sp500_score.empty else None
            sp500_pct = parse_score_to_pct(sp500_score_str) if sp500_score_str else 0

        rows_html = []
        for _, r in g.iterrows():
            metric = str(r["Metric"])
            if metric in ("Score", "—"):
                continue
            val = str(r["investment strategy"])
            sp500 = str(r["S&P 500"])
            rows_html.append(f"""
                <div class="metric-row">
                  <div class="metric-label">{escape(metric)}</div>
                  <div class="metric-value-portfolio">Portfolio: {escape(val)}</div>
                  <div class="metric-value-500">S&amp;P 500: {escape(sp500)}</div>
                </div>
            """)

        # Asset score 顯示進度條；其他 Category 顯示大字分數
        if cat == "Asset score":
            score_html = f'<span class="section-score">{escape(score_str)}</span>' if score_str else ""
            progress_html = f"""
              <div class="progress-track">
                <div class="progress-fill" style="width:{pct:.0f}%"></div>
                <div class="progress-fill-sp500" style="width:{sp500_pct:.0f}%;"></div>
              </div>
            """ if score_str and score_str not in ('—', '-') else ""
        else:
            score_html = f'<span class="section-score-large">{escape(score_str) if score_str else ""}</span>'
            progress_html = ""

        section = f"""
        <section class="section" aria-labelledby="{escape(cat).lower().replace(' ', '-')}-title">
          <header class="section-header">
            <h2 id="{escape(cat).lower().replace(' ', '-')}-title" class="section-title">{escape(cat)}</h2>
            {score_html}
          </header>
          {progress_html}
          <div class="metrics" role="list">
            {"".join(rows_html)}
          </div>
          
        </section>
        """
        sections_html.append(section)

    backtest_sections_html = "\n".join(sections_html)

    # HTML 模板
    html_template = """
    <!DOCTYPE html>
    <html lang="zh-Hant">
    <head>
        <meta charset="UTF-8">
        <title>投資組合報告</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { font-family: 'Helvetica Neue', Arial, sans-serif; margin: 20px; background-color: #0e0e10; color: #e0e0e0; }
            h1 { color: #ffffff; font-weight: 700; margin-bottom: 40px; }
            h2 { color: #b0b3b8; border-left: 4px solid #7b61ff; padding-left: 10px; margin-bottom: 20px; }
            h5 { color: #d1d1d1; margin-bottom: 15px; font-weight: 600; }
            .section { margin-bottom: 40px; padding-bottom: 20px; border-bottom: 1px solid #2a2a2d; }
            .flex-container { display: flex; flex-wrap: wrap; gap: 30px; justify-content: space-between; }
            .card-dark { background-color: #1c1c1f; border-radius: 12px; padding: 20px; flex: 1; min-width: 260px; box-shadow: 0 4px 10px rgba(0,0,0,0.5); }
            .card-dark img { max-width: 100%; height: auto; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.6); align-items: center;}
            /* render_backtest_html 樣式 */
            .section-header { display: flex; align-items: baseline; justify-content: space-between; gap: 16px; }
            .section-title { font-size: 18px; font-weight: 600; }
            .section-score { font-size: 22px; font-weight: 800; color: #008fc7; }
            .section-score-large { font-size: 22px; font-weight: 900; color: #ffffff; }
            .progress-track { margin-top: 10px; height: 8px; background: #2a2a2d; border-radius: 999px; overflow: hidden; }
            .progress-fill { height: 100%; background: linear-gradient(90deg, #4fc2f7, #008fc7); }
            .progress-fill-sp500 { height: 100%; background: linear-gradient(90deg, #ffa500, #ff8c00); }
            .metrics { margin-top: 6px; }
            .metric-row { display: flex; justify-content: space-between; padding: 8px 0; }
            .metric-label { color: #a1a1a1; text-align: left; flex: 2; padding-right: 1px; }
            .metric-value-500 { flex: 2; text-align: right; color: #ffffff; }
            .metric-value-portfolio { flex: 2; text-align: center; color: #ffffff; }
            /* 讓 df_stocks 和 df_intro 表格欄位與欄位值置中 */
            .table {
                border-collapse: collapse;
                border: 1px solid #444;
                width: 100%;
                table-layout: fixed;   /* 🔥 關鍵，強制依 CSS 分配欄寬 */
            }

            .table th, .table td {
                text-align: center !important;
                vertical-align: middle;
                border: 1px solid #ffffff; 
                padding-top: 8px;    /* 上方間距 */
                padding-bottom: 8px; 
                word-wrap: break-word;  /* 避免長字撐破表格 */
            }

        </style>
    </head>
    <body class="container">

        <div class="text-center">
            <h1>🚀 個人化投資組合分析報告</h1>
        </div>

        <div class="section">
            <h2> 📋 投組簡介</h2>
            <div class="flex-container">
                <div class="card-dark">
                    <div style="font-size: 16px;">
                    <br>親愛的{{ name }}客戶您好： </br>
                    <br>{{ intro_text }} </br>
                    </div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2> 📋 個股清單</h2>
            {{ stocks_table }}
        </div>

        <div class="section">
            <h2>📈 投資組合績效回測</h2>
            <div class="flex-container">
                <div class="card-dark">
                    <h5>績效折線圖</h5>
                    <img src="{{ plot_url }}" alt="Portfolio Performance Chart">
                </div>
                <div class="card-dark">
                    <h5>績效回測統計數據</h5>
                    {{ backtest_sections_html|safe }}
                    <div class="metric-label">💡備註：S&P 500各項分數皆為1.00x</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>📃 個股簡介</h2>
            {{ intro_table }}
        </div>

    </body>
    </html>
    """

    template = Template(html_template)
    rendered_html = template.render(
        stocks_table=stocks_table_html,
        intro_table=intro_table_html,
        plot_url=plot_url,
        backtest_sections_html=backtest_sections_html,
        intro_text = intro_text,
        name = name
    )
    return rendered_html
