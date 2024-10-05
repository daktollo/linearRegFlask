from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def ana_sayfa():
    return render_template("ana_sayfa.html")


app.run(debug=True, port=5789)