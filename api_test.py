from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/', methods=['POST'])
def echo():
    data = request.get_json()
    print("Received data:", data)
    return jsonify({"received": data})

if __name__ == '__main__':
    app.run()