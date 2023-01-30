import sys
import time
import json
import requests

def searchGlycoCT(seq, delay=0.5, maxretry=100):
    
    if type(seq) != list:
        seq=[seq]
    main_url = "https://edwardslab.bmcb.georgetown.edu/glylookup/"
    # main_url = "http://localhost:10980/"

    params = {
        "q": json.dumps(seq),
    }

    try:
        response1 = requests.post(main_url + "submit", params)
        response_json = response1.json()
        list_ids = list(response_json.values())
    except Exception as e:
        sys.stdout.write("Error: has issue connecting to flask API.")
        sys.stdout.write(str(e))
        sys.exit()

    # the list id for retrieval later
    #print (list_ids,"list_ids here")
    #print ("\n" * 3)

    params = {"q": json.dumps(list_ids)}

    # might require larger wait time in case you send a huge amounts of GlycoCTs
    for i in range(maxretry):

        time.sleep(delay)

        response2 = requests.post(main_url+ "retrieve", params)
        results = response2.json()
        #print("results here",results)

        for list_id, res in results.items():
            #print(f"list_id:{list_id}")
            #print(res['result']['error'])
            #print("res:", res["result"]['hits'])
            if res['finished']==True and res["result"]['hits']!=[]:
                accession=res['result']['hits'][0]
                return accession
            else:
                continue
            #print(f"hits:{res['result']['hits'][0]}")
            #print(f"entire resposne:{res['result']}")
            #for k, v in res.items():
            #    print(f"k:{k} v:{v}")
            #print ("- " * 20)

    return "Not found"



if __name__ == "__main__":
    glycoct = """RES
1b:x-dglc-HEX-x:x
2s:n-acetyl
3b:b-dglc-HEX-1:5
4s:n-acetyl
5b:b-dman-HEX-1:5
6b:a-dman-HEX-1:5
7b:a-dman-HEX-1:5
8b:a-dman-HEX-1:5
9b:a-dman-HEX-1:5
LIN
1:1d(2+1)2n
2:1o(4+1)3d
3:3d(2+1)4n
4:3o(4+1)5d
5:5o(3+1)6d
6:5o(6+1)7d
7:7o(3+1)8d
8:7o(6+1)9d"""

    print(searchGlycoCT(glycoct))
