#!/bin/env python3
#Reference: https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url

import requests, re, json, os

def download_fileids_from_google_drive(fid, destination_dir, extns, level=0):
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
        download_fileids_from_google_drive(dirs[k]['id'],dirs[k]['name'],extns,level=level+1)

    for k in sorted(files):
        f = files[k]
        extn = f['name'].rsplit('.',1)[-1]
        if extn not in extns:
            continue
        print("%s%s (%d bytes)..."%(" "*level*2,f['name'],f['size']),end=" ")
        sys.stdout.flush()
        os.makedirs(destination_dir,exist_ok=True)
        filepath = os.path.join(destination_dir,f['name'])
        if os.path.exists(filepath) and os.path.getsize(filepath) == f['size']:
            print("found.")
            continue
        download_file_from_google_drive(f['id'],filepath)
        print("downloaded.")

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
    if len(sys.argv) >= 2:
        dest_dir = sys.argv[1]
    if len(sys.argv) >= 3:
        folder_id = sys.argv[2]
    extensions = ("weights","labels","cfg")
    download_fileids_from_google_drive(folder_id, dest_dir, extensions)
