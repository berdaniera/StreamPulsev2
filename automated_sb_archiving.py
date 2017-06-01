import pysb
import config as cfg
import os
import time

## login to sb
sb = pysb.SbSession()
sb.login(cfg.SB_USER, cfg.SB_PASS)
time.sleep(5)

## check uploaded files against sb files
# metadata
metaf = os.listdir(cfg.META_FOLDER) # metadata files
insb = sb.get_item(cfg.SB_META) # get files
if len(insb)>0:
    fin_sb = [f['name'] for f in insb['files']] # get file names
    updata = [cfg.META_FOLDER+"/"+x for x in metaf if x not in fin_sb]
    metares = sb.upload_files_and_update_item(insb, upmeta)

time.sleep(2)

# original data
dataf = os.listdir(cfg.UPLOAD_FOLDER) # metadata files
insb = sb.get_item(cfg.SB_DATA) # get files
if len(insb)>0:
    fin_sb = [f['name'] for f in insb['files']] # get file names
    updata = [cfg.UPLOAD_FOLDER+"/"+x for x in dataf if x not in fin_sb]
    datares = sb.upload_files_and_update_item(insb, updata)
