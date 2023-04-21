import os
import re
import datetime
import inspect
import sys

def look():
    '''
    Prints basic info and loaded packages and verions
    
    Parts copied from https://stackoverflow.com/questions/38267791/how-to-i-list-imported-modules-with-their-version
    
    Hendrik Meuwese, november 2022

    Returns
    -------
    None.

    '''
    
    print('python '+sys.version)
    print("vandaag is ", datetime.date.today())
    print('environment '+os.environ['CONDA_DEFAULT_ENV']+' in folder '+os.environ['CONDA_PREFIX'])
    print('gebruiker '+os.environ['USR']+' op computer '+os.environ['COMPUTERNAME'])

    for name, val in sys._getframe(1).f_locals.items():
        if inspect.ismodule(val):

            fullnm = str(val)

            if not '(built-in)' in fullnm and \
               not __name__     in fullnm:
                m = re.search(r"'(.+)'.*'(.+)'", fullnm)
                module,path = m.groups()
                
                if hasattr(val, '__version__'):
                    str_version = f'version: {val.__version__:12s}'
                else :
                    str_version = f'version: geen nummer '
                print(f"%-12s {str_version} maps to %s" % (name, path))
