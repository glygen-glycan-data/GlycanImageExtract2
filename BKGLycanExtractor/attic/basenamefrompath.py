import ntpath

path = "test/p19-578.png"

basename = ntpath.basename(path).split('.')[0]

print(basename)