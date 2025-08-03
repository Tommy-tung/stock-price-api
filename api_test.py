from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)

@app.route('/read_excel', methods=['POST'])
def echo():
    # print(request)
    data = request.get_json()
    df = pd.read('s&p500_data.xlsx')
    stock_list = df['代碼'].values[:5]
    print("Received data:", data)
    return jsonify({"received": data, "stock" : stock_list})

if __name__ == '__main__':
    app.run()