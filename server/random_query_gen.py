import redis
import random

r = redis.Redis()
num_queries = 5000
max_query_len = 5

if not r.exists("WORDS"):
	for i in r.keys("PRESENT_IN:*"):
		word_id = i.split(":")[1]
		word = r.get("ID2WORD:%s" % word_id)
		r.sadd("WORDS", word)


for i in range(num_queries):
	n = random.randint(1, max_query_len)
	l = r.srandmember("WORDS", n)
	query = " ".join(l)
	print query
