
import inspect
import os.path

def callsig(back=0):
    fr = inspect.currentframe()
    for i in range(back+1):
        fr = fr.f_back
    func = fr.f_code
    self = fr.f_locals.get('self')
    clsname = ""
    if self is not None:
        clsname = self.__class__.__name__ + "."
    return "%s:%s %s%s"%(os.path.split(func.co_filename)[1],fr.f_lineno,clsname,func.co_name)
