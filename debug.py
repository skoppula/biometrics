DEBUG = True
LOG_TO_FILE = False
LOG_FILE = None

'''
    Prints to stdout (or log) iff we're in debug mode
    
    args:
        obj         the object to be printed
        num_tabs    the number of tabs to precede the obj printout
'''
def dpr(obj, num_tabs=0):
    if DEBUG:
        if not LOG_TO_FILE:
            with open(LOG_FILE, 'a') as f:
                f.write('\t'*num_tabs + str(obj))
        else:
            print '\t'*num_tabs + str(obj)

'''
    Prints to stdout (or log)

    args:
        obj         the object to be printed
        num_tabs    the number of tabs to precede the obj printout
'''
def pr(obj, num_tabs=0):
    if not LOG_TO_FILE:
        with open(LOG_FILE, 'a') as f:
            f.write('\t'*num_tabs + str(obj))
    else:
        print '\t'*num_tabs + str(obj)
