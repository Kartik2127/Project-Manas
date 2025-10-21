from flask import Flask, render_template

app = Flask(__name__,
            template_folder='frontend/templates',
            static_folder='frontend/static')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login')
def login():
    return render_template('login.html')

if __name__ == '__main__':
    app.run(debug=True)
