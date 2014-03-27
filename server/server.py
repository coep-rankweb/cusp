from flask import Flask, render_template, Request, request
import commands
from unidecode import unidecode

app = Flask(__name__)

def get_search_results(q):
	s = commands.getoutput("./mongo_sim2 %s" % q)
	try: s = unicode(s, "UTF-8")
	except: pass
	s = unidecode(s)
	d = []
	for i in s.split("\r\n"):
		tok = map(str.strip, i.split("\t"))
		try: d.append({'url': tok[0], 'title': tok[1]}) 
		except: pass
	return d

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/search')
def search():
	query = request.args.get('q')
	res = get_search_results(query)
	return render_template('search.html', results = res, query = query)

app.run(debug = True)
