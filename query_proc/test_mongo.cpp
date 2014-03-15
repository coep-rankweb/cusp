#include <cstdlib>
#include <iostream>
#include <mongo/client/dbclient.h>

using namespace mongo;
using namespace std;

int main() {
	DBClientConnection c;

	try {
		c.connect("localhost");
		BSONObj b = BSONObjBuilder().append("_id", 0).obj();
		auto_ptr<DBClientCursor> cursor = c.query("SPIDER_DB.URL_DATA", b);

		while (cursor->more()) {
			BSONObj p = cursor->next();
			//cout << p.getStringField("word_vec") << endl;
			cout << p["word_vec"] << endl;
		}
	}
	catch( const mongo::DBException &e ) {
		cout << "caught " << e.what() << endl;
	}

	return EXIT_SUCCESS;
}
