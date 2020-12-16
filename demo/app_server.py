from flask import Flask, render_template, request, jsonify
import json

from demo.core import toxicity

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route("/api/message", methods=['POST'])

def get_toxic():
    conv = json.loads(request.data)['conversation']
    print(conv)
    val = toxicity(conv)
    return jsonify({"toxicity" : val})

if __name__ == '__main__':
    app.run(debug=True)