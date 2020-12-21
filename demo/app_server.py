from flask import Flask, render_template, request, jsonify
import json

from demo.core2 import toxicity

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route("/api/message", methods=['POST'])
def get_toxic():
    request_data = json.loads(request.data)
    conv = request_data['conversation']
    rule = request_data['rule_text']
    print(conv)
    val = toxicity(conv, rule)
    return jsonify({"toxicity" : val})

if __name__ == '__main__':
    app.run(debug=True)