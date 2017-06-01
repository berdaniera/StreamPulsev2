import pysb
import config as cfg
import os
import time

## NEED TO ADD CODE FOR EMBARGOED SITES

## login to sb
sb = pysb.SbSession()
sb.login(cfg.SB_USER, cfg.SB_PASS)
time.sleep(5)

## check uploaded files against sb files
# metadata
metaf = os.listdir(cfg.META_FOLDER) # metadata files
insb = sb.get_item(cfg.SB_META) # get files
fin_sb = []
if 'files' in insb:
    fin_sb = [f['name'] for f in insb['files']] # get file names

upmeta = [cfg.META_FOLDER+"/"+x for x in metaf if x not in fin_sb]
print '\nSaving: '
print upmeta
metares = sb.upload_files_and_update_item(insb, upmeta)

time.sleep(2)

# original data
dataf = os.listdir(cfg.UPLOAD_FOLDER) # metadata files
insb = sb.get_item(cfg.SB_DATA) # get files
fin_sb = []
if 'files' in insb:
    fin_sb = [f['name'] for f in insb['files']] # get file names

updata = [cfg.UPLOAD_FOLDER+"/"+x for x in dataf if x not in fin_sb]
nper = 10
slices = [updata[x:x+nper] for x in xrange(0, len(updata), nper)]
for s in slices:
    print '\nSaving: '
    print s
    datares = sb.upload_files_and_update_item(insb, s)
    time.sleep(60)
