from jinja2 import Template
import pandas as pd
def generate_investment_report_html(df_stocks, df_backtest, df_intro):

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
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            h2 { color: #2c3e50; }
            table { border-collapse: collapse; width: 100%; margin-bottom: 40px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
            th { background-color: #f2f2f2; }
            .section { margin-bottom: 60px; }
        </style>
    </head>
    <body>

        <div class="section">
            <h2>📌 個股清單</h2>
            {{ stocks_table }}
        </div>

        <div class="section">
            <h2>📈 投資組合績效回測</h2>
            {{ backtest_table }}
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
    )

    return rendered_html
