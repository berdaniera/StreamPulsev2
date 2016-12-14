# -*- coding: utf-8 -*-
from flask import Flask, Markup, session, flash, render_template, request, jsonify, url_for, make_response, redirect, g
from flask_login import LoginManager, login_user, logout_user, current_user, login_required
from werkzeug.security import generate_password_hash, check_password_hash
from sunrise_sunset import SunriseSunset as suns
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
from dateutil import parser as dtparse
from math import log, sqrt, floor
import simplejson as json
import pandas as pd
import numpy as np
import requests
import binascii
import config as cfg
import pysb
import os
import re

app = Flask(__name__)
app.config['SECRET_KEY'] = cfg.SECRET_KEY
app.config['SQLALCHEMY_DATABASE_URI'] = cfg.SQLALCHEMY_DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = cfg.SQLALCHEMY_TRACK_MODIFICATIONS
app.config['UPLOAD_FOLDER'] = cfg.UPLOAD_FOLDER

#sb.login(cfg.SB_USER,cfg.SB_PASS)
#sbupf = sb.get_item(cfg.SB_UPFL)
########## DATABASE
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class Data(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    region = db.Column(db.Text)
    site = db.Column(db.Text)
    DateTime_UTC = db.Column(db.DateTime)
    variable = db.Column(db.Text)
    value = db.Column(db.Float)
    flag = db.Column(db.Integer)
    def __init__(self, region, site, DateTime_UTC, variable, value, flag):
        self.region = region
        self.site = site
        self.DateTime_UTC = DateTime_UTC
        self.variable = variable
        self.value = value
        self.flag = flag
    def __repr__(self):
        return '<Data %r, %r, %r, %r>' % (self.region, self.site, self.DateTime_UTC, self.variable)

class Flag(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    region = db.Column(db.Text)
    site = db.Column(db.Text)
    startDate = db.Column(db.DateTime)
    endDate = db.Column(db.DateTime)
    variable = db.Column(db.Text)
    flag = db.Column(db.Text)
    comment = db.Column(db.Text)
    by = db.Column(db.Integer) # user ID
    def __init__(self, region, site, startDate, endDate, variable, flag, comment, by):
        self.region = region
        self.site = site
        self.startDate = startDate
        self.endDate = endDate
        self.variable = variable
        self.flag = flag
        self.comment = comment
        self.by = by
    def __repr__(self):
        return '<Flag %r, %r, %r>' % (self.flag, self.comment, self.startDate)

class Tag(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    region = db.Column(db.Text)
    site = db.Column(db.Text)
    startDate = db.Column(db.DateTime)
    endDate = db.Column(db.DateTime)
    variable = db.Column(db.Text)
    tag = db.Column(db.Text)
    comment = db.Column(db.Text)
    by = db.Column(db.Integer) # user ID
    def __init__(self, region, site, startDate, endDate, variable, tag, comment, by):
        self.region = region
        self.site = site
        self.startDate = startDate
        self.endDate = endDate
        self.variable = variable
        self.tag = tag
        self.comment = comment
        self.by = by
    def __repr__(self):
        return '<Tag %r, %r>' % (self.tag, self.comment)

class Site(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    region = db.Column(db.Text)
    site = db.Column(db.Text)
    name = db.Column(db.Text)
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    usgs = db.Column(db.Text)
    addDate = db.Column(db.DateTime)
    embargo = db.Column(db.Boolean)
    by = db.Column(db.Integer)
    def __init__(self, region, site, name, latitude, longitude, usgs, addDate, embargo, by):
        self.region = region
        self.site = site
        self.name = name
        self.latitude = latitude
        self.longitude = longitude
        self.usgs = usgs
        self.addDate = addDate
        self.embargo = embargo
        self.by = by
    def __repr__(self):
        return '<Site %r, %r>' % (self.region, self.site)

class Cols(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    region = db.Column(db.Text)
    site = db.Column(db.Text)
    rawcol = db.Column(db.Text)
    dbcol = db.Column(db.Text)
    def __init__(self, region, site, rawcol, dbcol):
        self.region = region
        self.site = site
        self.rawcol = rawcol
        self.dbcol = dbcol
    def __repr__(self):
        return '<Cols %r, %r, %r>' % (self.region, self.site, self.dbcol)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(55), unique=True, index=True)
    password = db.Column(db.String(255))
    token = db.Column(db.String(100), nullable=False, server_default='')
    email = db.Column(db.String(255), unique=True)
    registered_on = db.Column(db.DateTime())
    confirmed = db.Column(db.Boolean)
    qaqc = db.Column(db.Text) # which qaqc sites can they save, comma separated?
    def __init__(self, username, password, email):
        self.username = username
        self.set_password(password)
        self.token = binascii.hexlify(os.urandom(10))
        self.email = email
        self.registered_on = datetime.utcnow()
        self.confirmed = True
        self.qaqc = ""
    def set_password(self, password):
        self.password = generate_password_hash(password)
    def check_password(self, password):
        return check_password_hash(self.password, password)
    def is_authenticated(self):
        return True
    def is_active(self):
        return True
    def is_anonymous(self):
        return False
    def get_id(self):
        return unicode(self.id)
    def qaqc_auth(self):
        return self.qaqc.split(",") # which tables can they edit
    def __repr__(self):
        return '<User %r>' % self.username

db.create_all()

@login_manager.user_loader
def load_user(id):
    return User.query.get(int(id))

@app.before_request
def before_request():
    g.user = current_user

###########
# #mysqldb eg
# cur = db.connection.cursor()
# cur.execute("SELECT * FROM articles")
# cur.execute("select distinct concat(region,'_',site) from data")
# x = cur.fetchall()
####################
########### FUNCTIONS
# Load core data sites
core = pd.read_csv('static/sitelist.csv')
core['SITECD'] = list(core["REGIONID"].map(str) +"_"+ core["SITEID"])
core = core.set_index('SITECD')
core.head()

# Get sunrise and sunset in UTC
# ro = suns(datetime.now(), latitude=core.LAT[12],longitude=core.LNG[12])
# rise_time, set_time = ro.calculate()
# datetime.strftime(rise_time,"%Y-%m-%d %H:%M:%S")
# datetime.strftime(set_time,"%Y-%m-%d %H:%M:%S")

# Need to make pickl with dictionary for each site - cache values, update selected value
# allsite = pd.read_csv("/home/aaron/spv2/static/SPsites.csv")
# allsite.head()
# # Save a dictionary into a pickle file.
# import pickle
# favorite_color = { "lion": "yellow", "kitty": "red" }
# pickle.dump( favorite_color, open( "save.p", "wb" ) )
# # reload it
# favorite_color = pickle.load( open( "save.p", "rb" ) )


variables = ['DateTime_UTC',
'DO_mgL',
'satDO_mgL',
'DOsat_pct',
'WaterTemp_C',
'WaterPres_kPa',
'AirTemp_C',
'AirPres_kPa',
'Depth_m',
'Discharge_m3s',
'Discharge_f3s',
'Velocity_ms',
'pH',
'CDOM_ugL',
'fDOM_mV',
'fDOM_frac',
'Turbidity_mV',
'Turbidity_NTU',
'Nitrate_mgL',
'SpecCond_mScm',
'SpecCond_uScm',
'Light_lux',
'Light_PAR',
'CO2_ppm']

# File uploading function
ALLOWED_EXTENSIONS = set(['txt', 'dat', 'csv'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def read_hobo(f):
    xt = pd.read_csv(f, skiprows=[0])
    cols = [x for x in xt.columns.tolist() if re.match("^#$|Coupler|File|Host|Connected|Attached|Stopped|End",x) is None]
    xt = xt[cols]
    m = [re.sub(" ","",x.split(",")[0]) for x in xt.columns.tolist()]
    u = [x.split(",")[1].split(" ")[1] for x in xt.columns.tolist()]
    tzoff = re.sub("GMT(.[0-9]{2}):([0-9]{2})","\\1",u[0])
    # if "-" in u[0]:
    #     tzoff = "+"+re.sub("GMT.([0-9]{2}):([0-9]{2})","\\1",u[0])
    # else:
    uu = [re.sub("\\ |\\/|°","",x) for x in u[1:]]
    uu = [re.sub(r'[^\x00-\x7f]',r'', x) for x in uu] # get rid of unicode
    newcols = ['DateTime']+[nme+unit for unit,nme in zip(uu,m[1:])]
    xt = xt.rename(columns=dict(zip(xt.columns.tolist(),newcols)))
    xt['DateTimeUTC'] = [dtparse.parse(x)-timedelta(hours=int(tzoff)) for x in xt.DateTime]
    if "_HW" in f:
        xt = xt.rename(columns={'AbsPreskPa':'water_kPa','TempC':'water_temp'})
    if "_HA" in f:
        xt = xt.rename(columns={'AbsPreskPa':'air_kPa','TempC':'air_temp'})
    if "_HD" in f:
        xt = xt.rename(columns={'TempC':'DO_temp'})
    if "_HP" in f:
        xt = xt.rename(columns={'TempC':'light_temp'})
    cols = xt.columns.tolist()
    return xt[cols[-1:]+cols[1:-1]]

def read_csci(f, gmtoff):
    xt = pd.read_csv(f, header=0, skiprows=[0,2,3])
    xt['DateTimeUTC'] = [dtparse.parse(x)-timedelta(hours=gmtoff) for x in xt.TIMESTAMP]
    cols = xt.columns.tolist()
    return xt[cols[-1:]+cols[1:-1]]

def read_manta(f, gmtoff):
    xt = pd.read_csv(f, skiprows=[0])
    xt = xt[~xt.DATE.str.contains('Eureka|DATE')]
    xt['DateTime'] = xt['DATE']+" "+xt['TIME']
    xt['DateTimeUTC'] = [dtparse.parse(x)-timedelta(hours=gmtoff) for x in xt.DateTime]
    xt.drop(["DATE","TIME","DateTime"], axis=1, inplace=True)
    xt = xt[[x for x in xt.columns.tolist() if " ." not in x and x!=" "]]
    xt.columns = [re.sub("\/|%","",x) for x in xt.columns.tolist()]
    splitcols = [x.split(" ") for x in xt.columns.tolist()]
    xt.columns = [x[0]+"_"+x[-1] if len(x)>1 else x[0] for x in splitcols]
    cols = xt.columns.tolist()
    return xt[cols[-1:]+cols[1:-1]]

def load_file(f, gmtoff, logger):
    if logger=="CS":
        return read_csci(f, gmtoff)
    elif "H" in logger:
        return read_hobo(f)
    elif logger=="EM":
        return read_manta(f, gmtoff)
    else:
        xtmp = pd.read_csv(f)
        xtmp = xtmp.rename(columns={xtmp.columns.values[0]:'DateTimeUTC'})
        return xtmp

def load_multi_file(ff, gmtoff, logger):
    f = [fi for fi in ff if "_"+logger in fi]
    if len(f)>1:
        xx = map(lambda x: load_file(x, gmtoff, logger), f)
        xx = reduce(lambda x,y: x.append(y), xx)
    else: # only one file for the logger, load it
        xx = load_file(f[0], gmtoff, logger)
    return wash_ts(xx)

# gmtoff = -7
# ff = ["/home/aaron/Downloads/AZ_OC_2016-11-14_EM.csv"]
# xx = load_file(ff[0],gmtoff,re.sub("(.*_)(.*)\\..*", "\\2", ff[0]))
# wash_ts(xx)
# xx.head()

# read and munge files for a site and date
def sp_in(ff, gmtoff): # ff must be a list!!!
    if len(ff)==1: # only one file, load
        xx = load_file(ff[0], gmtoff, re.sub("(.*_)(.*)\\..*", "\\2", ff[0]))
        xx = wash_ts(xx)
    else: # list by logger
        logger = list(set([re.sub("(.*_)(.*)\\..*", "\\2", f) for f in ff]))
        # if multiple loggers, map over loggers
        if len(logger)>1:
            xx = map(lambda x: load_multi_file(ff, gmtoff, x), logger)
            xx = reduce(lambda x,y: x.merge(y,how='outer',left_index=True,right_index=True), xx)
        else:
            logger = logger[0]
            xx = load_multi_file(ff, gmtoff, logger)
    return xx.reset_index()

def sp_in_lev(ff):
    xx = pd.read_csv(ff)
    return wash_ts(xx)

def wash_ts(x):
    cx = list(x.select_dtypes(include=['datetime64']).columns)[0]
    if x.columns.tolist()[0] != cx: # move date time to first column
        x = x[[cx]+[xo for xo in x.columns.tolist() if xo!=cx]]
    x = x.rename(columns={x.columns.tolist()[0]:'DateTime_UTC'})
    x = x.set_index("DateTime_UTC").sort_index().apply(lambda x: pd.to_numeric(x, errors='coerce')).resample('15Min').mean()
    return x

########## PAGES
@app.route('/register' , methods=['GET','POST'])
def register():
    if request.method == 'GET':
        return render_template('register.html')
    user = User(request.form['username'], request.form['password'], request.form['email'])
    db.session.add(user)
    db.session.commit()
    flash('User successfully registered', 'alert-success')
    return redirect(url_for('login'))

@app.route('/login',methods=['GET','POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    username = request.form['username']
    password = request.form['password']
    registered_user = User.query.filter_by(username=username).first()
    if registered_user is None:
        flash('Username is invalid' , 'alert-danger')
        return redirect(url_for('login'))
    if not registered_user.check_password(password):
        flash('Password is invalid', 'alert-danger')
        return redirect(url_for('login'))
    login_user(registered_user)
    flash('Logged in successfully', 'alert-success')
    return redirect(request.args.get('next') or url_for('index'))

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/')
@app.route('/index')
def index():
    nobs = pd.read_sql("select region, site, count(id) as n from data group by region, site", db.engine)
    nuse = pd.read_sql("select count(id) as n from user", db.engine)
    nobs = pd.read_sql("select count(id) as n from data", db.engine)
    return render_template('index.html',nobs="{:,}".format(nobs.n.sum()),nuse=nuse.n[0],nmod=0)

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':  # checks
        if 'file' not in request.files:
            flash('No file part','alert-danger')
            return redirect(request.url)
        ufiles = request.files.getlist("file")
        ufnms = [x.filename for x in ufiles]
        ffregex = "[A-Z]{2}_.*_[0-9]{4}-[0-9]{2}-[0-9]{2}_[A-Z]{2}.[a-zA-Z]{3}" # core sites
        ffregex2 = "[A-Z]{2}_.*_[0-9]{4}-[0-9]{2}-[0-9]{2}.csv" # leveraged sites
        pattern = re.compile(ffregex+"|"+ffregex2)
        if not all([pattern.match(f) is not None for f in ufnms]):
            # file names do not match expected pattern
            flash('Please name your files with the specified format.','alert-danger')
            return redirect(request.url)
        if all([f in os.listdir(app.config['UPLOAD_FOLDER']) for f in ufnms]):
            # all files already uploaded
            flash('All of those files were already uploaded.','alert-danger')
            return redirect(request.url)
        if (any([f in os.listdir(app.config['UPLOAD_FOLDER']) for f in ufnms])):
            # remove files already uploaded
            ufiles = [f for f in ufiles if f not in os.listdir(app.config['UPLOAD_FOLDER'])]
        site = list(set([x.split("_")[0]+"_"+x.split("_")[1] for x in ufnms]))
        if len(site)>1:
            flash('Please only select data from a single site.','alert-danger')
            return redirect(request.url)
        # UPLOAD locally and to sb
        filenames = []
        fnlong = []
        for file in ufiles:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                fup = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(fup)
                # sb.upload_file_to_item(sbupf, fup)
                filenames.append(filename)
                fnlong.append(fup)
        # PROCESS
        try:
            if site[0] in core.index.tolist():
                gmtoff = core.loc[site].GMTOFF[0]
                x = sp_in(fnlong, gmtoff)
            else:
                x = sp_in_lev(fnlong)
            tmp_file = site[0].encode('ascii')+"_"+binascii.hexlify(os.urandom(6))
            out_file = os.path.join(app.config['UPLOAD_FOLDER'],tmp_file+".csv")
            x.to_csv(out_file,index=False)
            columns = x.columns.tolist()
            cdict = pd.read_sql("select * from cols where concat(region,'_',site)='"+site[0]+"'", db.engine)
            cdict = dict(zip(cdict['rawcol'],cdict['dbcol']))
        except IOError:
            flash('Unknown error. Please email Aaron...','alert-danger')
            [os.remove(f) for f in fnlong]
            return redirect(request.url)
        return render_template('upload_columns.html', filenames=filenames, columns=columns, tmpfile=tmp_file, variables=variables, cdict=cdict)
    return render_template('upload.html')

@app.route("/upload_cancel",methods=["POST"])
def cancelcolumns():
    ofiles = request.form['ofiles'].split(",")
    tmpfile = request.form['tmpfile']+".csv"
    ofiles.append(tmpfile)
    [os.remove(os.path.join(app.config['UPLOAD_FOLDER'],x)) for x in ofiles] # remove tmp files
    flash('Upload cancelled.','alert-primary')
    return redirect(url_for('upload'))

def updatecdict(region, site, cdict):
    rawcols = pd.read_sql("select * from cols where region='"+region+"' and site ='"+site+"'", db.engine)
    rawcols = rawcols['rawcol'].tolist()
    for c in cdict.keys():
        if c in rawcols: # update
            cx = Cols.query.filter_by(rawcol=c).first()
            cx.dbcol = cdict[c] # assign column to value
            db.session.commit()
        else: # add
            cx = Cols(region, site, c, cdict[c])
            db.session.add(cx)
            db.session.commit()

@app.route("/upload_confirm",methods=["POST"]) # confirm columns
def confirmcolumns():
    cdict = json.loads(request.form['cdict'])
    tmpfile = request.form['tmpfile']
    cdict = dict([(r['name'],r['value']) for r in cdict])
    try: #something successful
        xx = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'],tmpfile+".csv"), parse_dates=[0])
        xx = xx[cdict.keys()].rename(columns=cdict)
        region, site = tmpfile.split("_")[:-1]
        xx = xx.set_index("DateTime_UTC")
        xx.columns.name = 'variable'
        xx = xx.stack()
        xx.name="value"
        xx = xx.reset_index()
        xx['region'] = region
        xx['site'] = site
        xx['flag'] = None
        xx = xx[['region','site','DateTime_UTC','variable','value','flag']]
        # add a check for duplicates?
        xx.to_sql("data", db.engine, if_exists='append', index=False)
        updatecdict(region, site, cdict)
    except IOError:
        flash('There was an error, please try again.','alert-warning')
        return redirect(request.url)
    os.remove(os.path.join(app.config['UPLOAD_FOLDER'],tmpfile+".csv")) # remove tmp file
    flash('Uploaded '+str(len(xx.index))+' values, thank you!','alert-success')
    return redirect(url_for('upload'))

# check login status...
@app.route('/download')
@login_required
def download():
    xx = pd.read_sql("select distinct concat(region,'_',site) as sites from data", db.engine)
    sites = xx['sites'].tolist()
    # sites = [(x.split("_")[0],x) for x in xx.sites.tolist()]
    # sitedict = {'0ALL':'ALL'}
    # for x in sites:
    #     if x[0] not in sitedict:
    #         sitedict[x[0]] = []
    #     sitedict[x[0]].append(x[1])
    return render_template('download.html',sites=sites)

@app.route('/_getstats',methods=['POST'])
def getstats():
    sitenm = request.json['site']
    xx = pd.read_sql("select * from data where concat(region,'_',site) in ('"+"', '".join(sitenm)+"') and flag is NULL", db.engine)
    startDate = xx.DateTime_UTC.min().strftime("%Y-%m-%d")
    endDate = (xx.DateTime_UTC.max()+timedelta(days=1)).strftime("%Y-%m-%d")
    initDate = (xx.DateTime_UTC.max()-timedelta(days=13)).strftime("%Y-%m-%d")
    #xx = pd.read_sql("select distinct variable from data", db.engine)
    variables = list(set(xx.variable))#xx['variable'].tolist()
    return jsonify(result="Success", startDate=startDate, endDate=endDate, initDate=initDate, variables=variables, site=sitenm)

@app.route('/_getcsv',methods=["POST"])
def getcsv():
    sitenm = request.form['site'].split(",")
    startDate = request.form['startDate']#.split("T")[0]
    endDate = request.form['endDate']
    variables = request.form['variables'].split(",")
    sqlq = "select * from data where concat(region,'_',site) in ('"+"', '".join(sitenm)+"') "+\
        "and DateTime_UTC>'"+startDate+"' "+\
        "and DateTime_UTC<'"+endDate+"' "+\
        "and variable in ('"+"', '".join(variables)+"')"
    xx = pd.read_sql(sqlq, db.engine)
    xx.loc[xx.flag==0,"value"] = None # set NA values
    xx.drop(['id','flag'], axis=1, inplace=True)
    resp = make_response(xx.to_csv(index=False))
    resp.headers["Content-Disposition"] = "attachment; filename=export.csv"
    resp.headers["Content-Type"] = "text/csv"
    return resp

@app.route('/visualize')
@login_required
def visualize():
    xx = pd.read_sql("select distinct concat(region,'_',site) as sites from data", db.engine)
    sites = xx['sites'].tolist()
    return render_template('visualize.html',sites=sites)

@app.route('/_getviz',methods=["POST"])
def getviz():
    sitenm = request.json['site'].split(",")
    startDate = request.json['startDate']
    endDate = request.json['endDate']#.split("T")[0]
    variables = request.json['variables']
    print sitenm, startDate, endDate, variables
    sqlq = "select * from data where concat(region,'_',site) in ('"+"', '".join(sitenm)+"') "+\
        "and DateTime_UTC>'"+startDate+"' "+\
        "and DateTime_UTC<'"+endDate+"' "+\
        "and variable in ('"+"', '".join(variables)+"')"
    xx = pd.read_sql(sqlq, db.engine)
    xx.loc[xx.flag==0,"value"] = None # set NaNs
    xx = xx.drop('id', axis=1).drop_duplicates()\
      .set_index(["DateTime_UTC","variable"])\
      .drop(['region','site','flag'],axis=1)\
      .unstack('variable')
    xx.columns = xx.columns.droplevel()
    xx = xx.reset_index()
    return jsonify(variables=variables, dat=xx.to_json(orient='records',date_format='iso'))

@app.route('/clean')
@login_required
def qaqc():
    xx = pd.read_sql("select distinct concat(region,'_',site) as sites from data", db.engine)
    qaqcuser = current_user.qaqc_auth()
    sites = [z for z in xx['sites'].tolist() if z in qaqcuser]
    xx = pd.read_sql("select distinct flag from flag", db.engine)
    flags = xx['flag'].tolist()
    return render_template('qaqc.html',sites=sites,flags=flags)

@app.route('/_getqaqc',methods=["POST"])
def getqaqc():
    sitenm = request.json['site']
    sqlq = "select * from data where concat(region,'_',site) = '"+sitenm+"'"
    xx = pd.read_sql(sqlq, db.engine)
    xx.loc[xx.flag==0,"value"] = None # set NaNs
    xx.dropna(subset=['value'], inplace=True) # remove rows with NA value
    variables = list(set(xx['variable'].tolist()))
    xx = xx.drop('id', axis=1).drop_duplicates()\
      .set_index(["DateTime_UTC","variable"])\
      .drop(['region','site','flag'],axis=1)\
      .unstack('variable')
    xx.columns = xx.columns.droplevel()
    xx = xx.reset_index()
    return jsonify(variables=variables, dat=xx.to_json(orient='records',date_format='iso'))

@app.route('/_addflag',methods=["POST"])
def addflag():
    rgn, ste = request.json['site'].split("_")
    sdt = dtparse.parse(request.json['startDate'])
    edt = dtparse.parse(request.json['endDate'])
    var = request.json['var']
    flg = request.json['flagid']
    cmt = request.json['comment']
    fff = Flag(rgn, ste, sdt, edt, var, flg, cmt, int(current_user.get_id()))
    db.session.add(fff)
    db.session.commit()
    flgdat = Data.query.filter(Data.region==rgn,Data.site==ste,Data.DateTime_UTC>=sdt,Data.DateTime_UTC<=edt,Data.variable==var).all()
    for f in flgdat:
        f.flag = fff.id
    db.session.commit()
    return jsonify(result="success")

@app.route('/_addtag',methods=["POST"])
def addtag():
    rgn, ste = request.json['site'].split("_")
    sdt = dtparse.parse(request.json['startDate'])
    edt = dtparse.parse(request.json['endDate'])
    var = request.json['var']
    tag = request.json['tagid']
    cmt = request.json['comment']
    ttt = Tag(rgn, ste, sdt, edt, var, tag, cmt, int(current_user.get_id()))
    db.session.add(ttt)
    db.session.commit()
    # flgdat = Data.query.filter(Data.region==rgn,Data.site==ste,Data.DateTime_UTC>=sdt,Data.DateTime_UTC<=edt,Data.variable==var).all()
    # for f in flgdat:
    #     f.flag = fff.id
    # db.session.commit()
    return jsonify(result="success")

@app.route('/_addna',methods=["POST"])
def addna():
    rgn, ste = request.json['site'].split("_")
    sdt = dtparse.parse(request.json['startDate'])
    edt = dtparse.parse(request.json['endDate'])
    var = request.json['var']
    # add NA flag = 0
    flgdat = Data.query.filter(Data.region==rgn,Data.site==ste,Data.DateTime_UTC>=sdt,Data.DateTime_UTC<=edt,Data.variable==var).all()
    for f in flgdat:
        f.flag = 0
    db.session.commit()
    # new query
    sqlq = "select * from data where concat(region,'_',site) = '"+rgn+"_"+ste+"'"
    xx = pd.read_sql(sqlq, db.engine)
    xx.loc[xx.flag==0,"value"] = None # set NaNs
    xx.dropna(subset=['value'], inplace=True) # remove rows with NA value
    xx = xx.drop('id', axis=1).drop_duplicates()\
      .set_index(["DateTime_UTC","variable"])\
      .drop(['region','site','flag'],axis=1)\
      .unstack('variable')
    xx.columns = xx.columns.droplevel()
    xx = xx.reset_index()
    return jsonify(dat=xx.to_json(orient='records',date_format='iso'))

if __name__=='__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
