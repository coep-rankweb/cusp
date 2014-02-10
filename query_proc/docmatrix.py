import redis
import os
import sys

r = redis.Redis()

parent = "/home/nvdia/programs/Crawlers/config_data/classes_odp/"
categories = os.listdir(parent)

market = open(sys.argv[1], "w")
market.write("%%MatrixMarket matrix coordinate real general\n%\n\n")

doc_cnt = 1
for cat in categories:
	f = open(parent + cat + "/features.txt")
	for doc in f:
		l = []
		tokens = doc.strip().split(",")[:-1]
		for tok in tokens:
			val = r.get("FEATURE_COL:" + tok)
			if val: l.append(int(val))
		if l:
			for i in l:
				market.write("%d\t%d\t%lf\n" % (doc_cnt, i, 1 / len(l) ** 0.5))
			doc_cnt += 1
	print doc_cnt
	f.close()

market.close()
