import sys

try:
	f = open (sys.argv[1])
except IndexError:
	sys.exit(1)


s = sum(float(i) for i in f)
f.close ()

print s
