from flask import Flask, render_template, Request, request, make_response, Response
import commands
from unidecode import unidecode
import redis
import time
import random

QUERY_PIPE = "/home/nvidia/query_pipe"
RESULT_PIPE = "/home/nvidia/result_pipe"

app = Flask(__name__)

f = open(QUERY_PIPE, "w")
g = open(RESULT_PIPE ,"r")

r = redis.Redis()

'''
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
'''

def generate_id():
	return str(int(random.random() * time.time()) % 1000)

def fetch_results(query, _id, page = 1):
	cached_result = r.get("QCACHE:%s:%s" % (_id, query))
	if cached_result:
		res = eval(cached_result)
		return res[10 * (page - 1) : 10 * page], len(res)
	else:
		f.write(query + "\n")
		f.flush()
		s = g.read()
		if s:
			res = [dict(zip(["url", "title"], map(clean, i.split("<@@@>")))) for i in s.split("<###>")]
		else:
			return [], 0
		r.set("QCACHE:%s:%s" % (_id, query), str(res))
	return res[10 * (page - 1) : 10 * page], len(res)

def clean(s):
	try:
		s = unicode(s, 'UTF-8')
	except:
		pass
	s = s.strip()
	s = unidecode(s)
	return s

@app.route('/')
def search():
	query = request.args.get('q')
	if not query or not query.strip():
		return render_template('search.html', results = [], query = "", page = 1, more_results = False)
	page = int(request.args.get('page', 1))

	_id = request.cookies.get('_id', None)
	response = Response()
	if not _id:
		_id = generate_id()
		response.set_cookie('_id', _id)

	start = time.time()
	res, total_len = fetch_results(query, _id, page)
	end = time.time()
	r = render_template('search.html', results = res, query = query, page = int(page), num_results = total_len, more_results = (page * 10 < total_len), time = end - start)
	response.set_data(r)
	return response

app.run(debug = True, host = "0.0.0.0")
