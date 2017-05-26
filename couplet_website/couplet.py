import os, sys
import sqlite3
from flask import Flask, request, session, g, redirect, url_for, abort, \
     render_template, flash



from model import generate, select_random_sentences

app = Flask(__name__)

@app.route('/', methods = ['POST', 'GET'])
def index(data=""):
    error = None
    if request.method == 'POST':
        data = request.form['s'] 
        if len(data) == 0:
            shanglian = select_random_sentences('../data/train.txt')
            xialian = []        
            return render_template('index.html', shanglian = shanglian, xialian = xialian, data = data)         
        shanglian = select_random_sentences('../data/train.txt')
        print 'help'
        xialian = generate(data)
        return render_template('index.html', shanglian = shanglian, xialian = xialian, data = data)
    else:
        if len(data) == 0:
            shanglian = select_random_sentences('../data/train.txt')
            xialian = []        
            return render_template('index.html', shanglian = shanglian, xialian = xialian, data = data)         
        shanglian = select_random_sentences('../data/train.txt')
        xialian = generate(data)
        return render_template('index.html', shanglian = shanglian, xialian = xialian, data = data)

@app.context_processor
def override_url_for():
    return dict(url_for=dated_url_for)

def dated_url_for(endpoint, **values):
    if endpoint == 'static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path,
                                     endpoint, filename)
            values['q'] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
    #app.run(host='0.0.0.0')
