from jinja2 import Template
import pandas as pd
def generate_investment_report_html(df_stocks, df_backtest, df_intro, plot_url):

    # 轉成 HTML 表格
    stocks_table_html = df_stocks.to_html(index=False, classes="table")
    backtest_table_html = df_backtest.to_html(index=False, classes="table")
    intro_table_html = df_intro.to_html(index=False, classes="table")

    # HTML 模板
    html_template = """
    <!DOCTYPE html>
    <html lang="zh-Hant">
    <head>
        <meta charset="UTF-8">
        <title>投資組合報告</title>

        <!-- Bootstrap 5 CSS -->
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">

        <style>
            body { font-family: 'Helvetica Neue', Arial, sans-serif; margin: 40px; }
            h1 { color: #1a1a1a; margin-bottom: 50px; }
            h2 { color: #2c3e50; margin-top: 40px; }
            h5 { color: #495057; margin-top: 20px; margin-bottom: 15px; }
            table { border-collapse: collapse; width: 100%; margin-bottom: 40px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
            th { background-color: #f8f9fa; }
            .section { margin-bottom: 60px; }

            .flex-container {
                display: flex;
                flex-wrap: wrap;
                gap: 30px;
                justify-content: space-between;
            }

            .flex-item {
                flex: 1;
                min-width: 300px;
            }

            .flex-item img {
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }

            table {
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            }
        </style>
    </head>
    <body class="container">

        <!-- 頁面大標題 -->
        <div class="text-center">
            <h1 class="display-4">💼 個人化投資組合分析大屌報告</h1>
        </div>

        <div class="section">
            <h2>📌 個股清單</h2>
            {{ stocks_table }}
        </div>

        <div class="section">
            <h2>📈 投資組合績效回測</h2>
            <div class="flex-container">
                <div class="flex-item">
                    <h5>📊 績效折線圖</h5>
                    <img src="{{ plot_url }}" alt="Portfolio Performance Chart">
                </div>
                <div class="flex-item">
                    <h5>📋 績效回測統計數據</h5>
                    {{ backtest_table }}
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

    # 套用模板
    template = Template(html_template)
    rendered_html = template.render(
        stocks_table=stocks_table_html,
        backtest_table=backtest_table_html,
        intro_table=intro_table_html,
        plot_url=plot_url
    )

    return rendered_html
