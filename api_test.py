from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/', methods=['GET'])
def echo():
    # print(request)
    # data = request.get_json()
    # print("Received data:", request.data)
    # return jsonify({"received": request.data})
    return jsonify({"message": "Hello from updated API!"})

if __name__ == '__main__':
    app.run()