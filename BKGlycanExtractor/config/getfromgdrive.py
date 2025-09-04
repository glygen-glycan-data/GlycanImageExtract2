#!/bin/env python3
#Reference: https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url

import requests, re, json, os, glob, os.path

def download_fileids_from_google_drive(fid, destination_dir, path, configs, extns, rmlocal=False, level=0):

    if isinstance(configs,str):
        configfiles = set()
        with open(configs) as configfile:
            for l in configfile:
                if '=' not in l:
                    continue
                key,value = [ s.strip() for s in l.split('=',1) ]
                if key in ("weights","config"):
                     configfiles.add(value)
                     base = value.rsplit('.',1)[0]
                     if key == "weights":
                         configfiles.add(base + ".labels")
                         configfiles.add(base + ".model")
        configs = configfiles

    present = set()
    for extn in extns:
        for fn in glob.glob(os.path.join(destination_dir,path,"*."+extn)):
            if os.path.isfile(fn):
                present.add(os.path.join(path,os.path.split(fn)[1]))

    URL = "https://drive.google.com/drive/u/2/folders/"
    session = requests.Session()

    response = session.get(URL + fid)

    files = dict()
    dirs = dict()

    data = re.sub(r'(</[a-zA-Z]+>)',r'\1\n',response.text)
    for l in data.splitlines():
        if 'window[\'_DRIVE_ivd\'] = ' in l:
            l = eval(l.split('window[\'_DRIVE_ivd\'] = ',1)[1].split(';if ',1)[0].replace(r'\/','/'))
            data = json.loads(l)
            data = data[0]
            for i,it in enumerate(data):
                # print(it)
                itdata = dict(id=it[0],name=it[2],type=it[3],size=it[13],ordinal=i+1)
                if 'folder' in it[3]:
                    dirs[i] = itdata
                else:
                    files[i] = itdata

    for k in sorted(dirs):
        # print("%s%s/"%(" "*level*2,dirs[k]['name']))
        if path:
            newpath = os.path.join(path,dirs[k]['name'])
        else:
            newpath = dirs[k]['name']
        download_fileids_from_google_drive(dirs[k]['id'],destination_dir,newpath,configs,extns,rmlocal,level=level+1)

    for k in sorted(files):
        f = files[k]
        extn = f['name'].rsplit('.',1)[-1]
        if extn not in extns:
            continue
        filepath = os.path.join(path,f['name'])
        if filepath in present:
            present.remove(filepath)
        if filepath not in configs:
            # print("not needed.")
            continue
        print("%s (%d bytes)..."%(filepath,f['size']),end=" ")
        sys.stdout.flush()
        os.makedirs(os.path.join(destination_dir,path),exist_ok=True)
        filepath1 = os.path.join(destination_dir,filepath)
        if os.path.exists(filepath1) and os.path.getsize(filepath1) == f['size']:
            print("found.")
            continue
        download_file_from_google_drive(f['id'],filepath1)
        print("downloaded.")

    if rmlocal:
        for fn in present:
            os.unlink(os.path.join(destination_dir,fn))
            print("Local file",fn,"removed.")

def download_file_from_google_drive(id, destination):
    URL = "https://drive.usercontent.google.com/download"

    session = requests.Session()

    params = { 'id' : id, 'export': 'download' }
    # print(URL, params)
    response = session.get(URL, params = params, stream = True)
    token = get_confirm_token(response)

    if token:
        params['confirm'] = token
        # print(URL, params)
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    if "Virus scan warning" in response.text:
        return 't'
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

if __name__ == "__main__":

    import sys
    dest_dir = '.'
    folder_id = '1cK7xwAKl5jwezDBZRUDyYVltVHv1NsRf'
    configs = 'configs.ini'
    rmlocal = False
    
    if len(sys.argv) >= 2 and sys.argv[1] == "--clean":
        rmlocal = True
        sys.argv.pop(1)
    if len(sys.argv) >= 2:
        dest_dir = sys.argv[1]
    if len(sys.argv) >= 3:
        folder_id = sys.argv[2]
    if len(sys.argv) >= 4:
        configs = sys.argv[3]
    extensions = ("weights","labels","cfg","model")
    download_fileids_from_google_drive(folder_id, dest_dir, "", configs, extensions, rmlocal)
