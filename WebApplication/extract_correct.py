#!/bin/env python3.12

import json, sys

result = json.load(open(sys.argv[1]))

for k in list(result):
    if k not in ("result",):
        del result[k]

for k in list(result["result"]):
    if k not in ("figure_result",):
        del result["result"][k]

for f in result["result"]["figure_result"]:
    for k in list(f):
        if k not in ("glycans",):
            del f[k]
    for g in f["glycans"]:
        for k in list(g):
            if k not in ("IUPAC","composition_str","bbox","accession"):
                del g[k]

print(json.dumps(result,indent=2))

