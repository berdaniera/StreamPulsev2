# -*- coding: utf-8 -*-
from flask import Flask, Markup, session, flash, render_template, request, jsonify, url_for, make_response, send_file, redirect, g
from flask_login import LoginManager, login_user, logout_user, current_user, login_required
from werkzeug.security import generate_password_hash, check_password_hash
from sunrise_sunset import SunriseSunset as suns
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func
from itsdangerous import URLSafeTimedSerializer
from datetime import datetime, timedelta
from dateutil import parser as dtparse
from math import log, sqrt, floor
import simplejson as json
from sklearn import svm
import pandas as pd
import numpy as np
import requests
import binascii
import tempfile
import zipfile
import shutil
#import pysb
import os
import re
import config as cfg

app = Flask(__name__)
app.config['SECRET_KEY'] = cfg.SECRET_KEY
app.config['SQLALCHEMY_DATABASE_URI'] = cfg.SQLALCHEMY_DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = cfg.SQLALCHEMY_TRACK_MODIFICATIONS
app.config['UPLOAD_FOLDER'] = cfg.UPLOAD_FOLDER
app.config['META_FOLDER'] = cfg.META_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16 MB
app.config['SECURITY_PASSWORD_SALT'] = cfg.SECURITY_PASSWORD_SALT

#sb.login(cfg.SB_USER,cfg.SB_PASS)
#sbupf = sb.get_item(cfg.SB_UPFL)
########## DATABASE
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class Data(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    region = db.Column(db.String(10))
    site = db.Column(db.String(50))
    DateTime_UTC = db.Column(db.DateTime)
    variable = db.Column(db.String(50))
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

class Manual(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    region = db.Column(db.String(10))
    site = db.Column(db.String(50))
    DateTime_UTC = db.Column(db.DateTime)
    variable = db.Column(db.String(50))
    value = db.Column(db.Float)
    def __init__(self, region, site, DateTime_UTC, variable, value):
        self.region = region
        self.site = site
        self.DateTime_UTC = DateTime_UTC
        self.variable = variable
        self.value = value
    def __repr__(self):
        return '<Manual %r, %r, %r, %r>' % (self.region, self.site, self.DateTime_UTC, self.variable)

class Flag(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    region = db.Column(db.String(10))
    site = db.Column(db.String(50))
    startDate = db.Column(db.DateTime)
    endDate = db.Column(db.DateTime)
    variable = db.Column(db.String(50))
    flag = db.Column(db.String(50))
    comment = db.Column(db.String(255))
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
    region = db.Column(db.String(10))
    site = db.Column(db.String(50))
    startDate = db.Column(db.DateTime)
    endDate = db.Column(db.DateTime)
    variable = db.Column(db.String(50))
    tag = db.Column(db.String(50))
    comment = db.Column(db.String(255))
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
    region = db.Column(db.String(10))
    site = db.Column(db.String(50))
    name = db.Column(db.String(50))
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    usgs = db.Column(db.String(20))
    addDate = db.Column(db.DateTime)
    embargo = db.Column(db.Boolean)
    by = db.Column(db.Integer)
    contact = db.Column(db.String(50))
    contactEmail = db.Column(db.String(255))
    def __init__(self, region, site, name, latitude, longitude, usgs, addDate, embargo, by, contact, contactEmail):
        self.region = region
        self.site = site
        self.name = name
        self.latitude = latitude
        self.longitude = longitude
        self.usgs = usgs
        self.addDate = addDate
        self.embargo = embargo
        self.by = by
        self.contact = contact
        self.contactEmail = contactEmail
    def __repr__(self):
        return '<Site %r, %r>' % (self.region, self.site)

class Cols(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    region = db.Column(db.String(10))
    site = db.Column(db.String(50))
    rawcol = db.Column(db.String(100))
    dbcol = db.Column(db.String(100))
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
        self.confirmed = True # do they agree to the policy?
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

class Downloads(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime)
    userID = db.Column(db.Integer)
    email = db.Column(db.String(255))
    dnld_sites = db.Column(db.String(500))
    dnld_date0 = db.Column(db.DateTime)
    dnld_date1 = db.Column(db.DateTime)
    dnld_vars = db.Column(db.String(500))
    def __init__(self, timestamp, userID, email, dnld_sites, dnld_date0, dnld_date1, dnld_vars):
        self.timestamp = timestamp
        self.userID = userID
        self.email = email
        self.dnld_sites = dnld_sites
        self.dnld_date0 = dnld_date0
        self.dnld_date1 = dnld_date1
        self.dnld_vars = dnld_vars
    def __repr__(self):
        return '<Download %r>' % (self.id)

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

variables = ['DateTime_UTC',
'DO_mgL',
'satDO_mgL',
'DOsat_pct',
'WaterTemp_C',
'WaterPres_kPa',
'AirTemp_C',
'AirPres_kPa',
'Level_m',
'Depth_m',
'Discharge_m3s',
'Velocity_ms',
'pH',
'pH_mV',
'CDOM_ppb',
'CDOM_mV',
'Turbidity_NTU',
'Turbidity_mV',
'Nitrate_mgL',
'SpecCond_mScm',
'SpecCond_uScm',
'CO2_ppm',
'Light_lux',
'Light_PAR',
'Light2_lux',
'Light2_PAR',
'Light3_lux',
'Light3_PAR',
'Light4_lux',
'Light4_PAR',
'Light5_lux',
'Light5_PAR',
'Battery_V']

# File uploading function
ALLOWED_EXTENSIONS = set(['txt', 'dat', 'csv'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def read_hobo(f):
    xt = pd.read_csv(f, skiprows=[0])
    cols = [x for x in xt.columns.tolist() if re.match("^#$|Coupler|File|Host|Connected|Attached|Stopped|End|Unnamed|Good|Bad",x) is None]
    xt = xt[cols]
    m = [re.sub(" ","",x.split(",")[0]) for x in xt.columns.tolist()]
    u = [x.split(",")[1].split(" ")[1] for x in xt.columns.tolist()]
    tzoff = re.sub("GMT(.[0-9]{2}):([0-9]{2})","\\1",u[0])
    ll = f.split("_")[3].split(".")[0].split("v")[0]
    inum = re.sub("[A-Z]{2}","",ll)
    uu = [re.sub("\\ |\\/|°","",x) for x in u[1:]]
    uu = [re.sub(r'[^\x00-\x7f]',r'', x) for x in uu] # get rid of unicode
    newcols = ['DateTime']+[nme+unit for unit,nme in zip(uu,m[1:])]
    xt = xt.rename(columns=dict(zip(xt.columns.tolist(),newcols)))
    xt['DateTimeUTC'] = [dtparse.parse(x)-timedelta(hours=int(tzoff)) for x in xt.DateTime]
    # xt = xt.rename(columns={'HWAbsPreskPa':'WaterPres_kPa','HWTempC':'WaterTemp_C','HAAbsPreskPa':'AirPres_kPa','HATempC':'AirTemp_C',})
    if "_HW" in f:
        xt = xt.rename(columns={'AbsPreskPa':'WaterPres_kPa','TempC':'WaterTemp_C'})
    if "_HA" in f:
        xt = xt.rename(columns={'AbsPreskPa':'AirPres_kPa','TempC':'AirTemp_C'})
    if "_HD" in f:
        xt = xt.rename(columns={'TempC':'DOTemp_C'})
    if "_HP" in f:
        xt = xt.rename(columns={'TempC':'LightTemp_C'})
    xt.columns = [cc+inum for cc in xt.columns]
    cols = xt.columns.tolist()
    return xt[cols[-1:]+cols[1:-1]]

def read_csci(f, gmtoff):
    xt = pd.read_csv(f, header=0, skiprows=[0,2,3])
    xt['DateTimeUTC'] = [dtparse.parse(x)-timedelta(hours=gmtoff) for x in xt.TIMESTAMP]
    cols = xt.columns.tolist()
    return xt[cols[-1:]+cols[1:-1]]

def read_manta(f, gmtoff):
    xt = pd.read_csv(f, skiprows=[0])
    if 'Eureka' in xt.columns[0]:
        xt.columns = xt.loc[0].tolist()
    xt = xt[~xt.DATE.str.contains('Eureka|DATE')]
    xt['DateTime'] = xt['DATE']+" "+xt['TIME']
    xt['DateTimeUTC'] = [dtparse.parse(x)-timedelta(hours=gmtoff) for x in xt.DateTime]
    xt.drop(["DATE","TIME","DateTime"], axis=1, inplace=True)
    xt = xt[[x for x in xt.columns.tolist() if " ." not in x and x!=" "]]
    xt.columns = [re.sub("\/|%","",x) for x in xt.columns.tolist()]
    splitcols = [x.split(" ") for x in xt.columns.tolist()]
    xt.columns = [x[0]+"_"+x[-1] if len(x)>1 else x[0] for x in splitcols]
    cols = xt.columns.tolist()
    xt = xt.set_index('DateTimeUTC').apply(lambda x: pd.to_numeric(x, errors='coerce')).reset_index()
    return xt

def load_file(f, gmtoff, logger):
    if "CS" in logger:
        return read_csci(f, gmtoff)
    elif "H" in logger:
        return read_hobo(f)
    elif "EM" in logger:
        return read_manta(f, gmtoff)
    else:
        xtmp = pd.read_csv(f, parse_dates=[0])
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
    xx = pd.read_csv(ff, parse_dates=[0])
    return wash_ts(xx).reset_index()

def wash_ts(x):
    cx = list(x.select_dtypes(include=['datetime64']).columns)[0]
    if x.columns.tolist()[0] != cx: # move date time to first column
        x = x[[cx]+[xo for xo in x.columns.tolist() if xo!=cx]]
    x = x.rename(columns={x.columns.tolist()[0]:'DateTime_UTC'})
    x = x.set_index("DateTime_UTC").sort_index().apply(lambda x: pd.to_numeric(x, errors='coerce')).resample('15Min').mean().dropna(how='all')
    return x

def panda_usgs(x,jsof):
    ts = jsof['value']['timeSeries'][x]
    usgst = pd.read_json(json.dumps(ts['values'][0]['value']))
    vcode = ts['variable']['variableCode'][0]['value']
    if vcode=='00060': # discharge
        usgst.value = usgst.value/35.3147
        colnm = 'USGSDischarge_m3s'
    else:
        usgst.value = usgst.value/3.28084
        colnm = 'USGSLevel_m'
    usgst['site'] = ts['sourceInfo']['siteCode'][0]['value'] # site code
    return {ts['sourceInfo']['siteCode'][0]['value']:usgst[['dateTime','value']].rename(columns={'dateTime':'DateTime_UTC','value':colnm}).set_index(["DateTime_UTC"])}

def get_usgs(regionsite, startDate, endDate, vvv=['00060','00065']):
    # regionsite is a list
    # vvv is a list of variable codes
    #00060 is cfs, discharge; 00065 is feet, height
    xs = pd.read_sql("select * from site",db.engine)
    xs['regionsite'] = xs["region"].map(str)+"_"+xs["site"]
    # for each region site...
    sitex = xs.loc[xs.regionsite.isin(regionsite)].usgs.tolist()
    sitedict = dict(zip(sitex,regionsite))
    sitex = [x for x in sitex if x is not None]
    usgs = ",".join(sitex)
    #lat,lng = sitex.loc[:,['latitude','longitude']].values.tolist()[0]
    if(len(sitex)==0 or usgs is None):
        return []
    vcds = '00060,00065'#",".join(vvv)
    url = "https://nwis.waterservices.usgs.gov/nwis/iv/?format=json&sites="+usgs+"&startDT="+startDate+"&endDT="+endDate+"&parameterCd="+vcds+"&siteStatus=all"
    r = requests.get(url)
    r.status_code
    xf = r.json()
    xx = map(lambda x: panda_usgs(x, xf), range(len(xf['value']['timeSeries'])))
    xoo = []
    for s in sitex:
        x2 = [k.values()[0] for k in xx if k.keys()[0]==s]
        x2 = reduce(lambda x,y: x.merge(y,how='outer',left_index=True,right_index=True), x2)
        x2 = x2.sort_index().apply(lambda x: pd.to_numeric(x, errors='coerce')).resample('15Min').mean()
        x2['site']=sitedict[s]
        xoo.append(x2.reset_index())
    #
    xx = pd.concat(xoo)
    xx = xx.set_index(['DateTime_UTC','site'])
    xx.columns.name='variable'
    xx = xx.stack()
    xx.name="value"
    xx = xx.reset_index()
    xx[['region','site']] = xx['site'].str.split("_",expand=True)
    xx.head()
    return xx[['DateTime_UTC','region','site','variable','value']]

def authenticate_sites(sites,user=None,token=None):
    ss = []
    for site in sites:
        r,s = site.split("_")
        ss.append("(region='"+r+"' and site='"+s+"') ")
    qs = "or ".join(ss)
    xx = pd.read_sql("select region, site, embargo, site.by from site where "+qs, db.engine)
    if token is not None:
        tt = pd.read_sql("select * from user where token='"+token+"'", db.engine)
        if(len(tt)==1):
            user = str(tt.id[0])
    if user is not None: # return public sites and authenticated sites
        xx = xx[(xx['embargo']==0)|(xx['by']==int(user))]
    else: # return only public sites
        xx = xx[(xx['embargo']==0)] # return
    return [x[0]+"_"+x[1] for x in zip(xx.region,xx.site)]

def generate_confirmation_token(email):
    serializer = URLSafeTimedSerializer(app.config['SECRET_KEY'])
    return serializer.dumps(email, salt=app.config['SECURITY_PASSWORD_SALT'])

def confirm_token(token, expiration=3600*24): # expires in one day
    serializer = URLSafeTimedSerializer(app.config['SECRET_KEY'])
    try:
        email = serializer.loads(token,
            salt=app.config['SECURITY_PASSWORD_SALT'],
            max_age=expiration)
    except:
        return False
    return email

# def send_email(to, subject, template):
#     mail.send_message(subject, recipients=[to], html=template)

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

@app.route('/_reset_sp_pass', methods=['GET','POST'])
@login_required
def resetpass():
    if request.method == 'GET':
        return render_template('reset.html')
    email = request.form['email']
    try:
        user = User.query.filter(User.email==email).first_or_404()
    except: # email is not confirmed
        flash("We couldn't find an account with that email.", 'alert-danger')
        return redirect(url_for('resetpass'))
    token = generate_confirmation_token(email)
    register_url = url_for('resetpass_confirm', token=token, _external=True)
    return jsonify(account=email, reset_url=register_url, note="This link is valid for 24 hours.")

@app.route('/resetpass/<token>' , methods=['GET','POST'])
def resetpass_confirm(token):
    if request.method == 'GET':
        try:
            email = confirm_token(token)
            user = User.query.filter(User.email==email).first_or_404()
        except: # email is not confirmed
            flash('The confirmation link is invalid or has expired.', 'alert-danger')
            return redirect(url_for('index'))
        return render_template('resetpass.html', email=email) # email is good.
    # posting new password
    user = User.query.filter(User.email==request.form['email']).first_or_404()
    user.password = generate_password_hash(request.form['password'])
    db.session.add(user)
    db.session.commit()
    flash('Successfully reset, please login.', 'alert-success')
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
    nuse = pd.read_sql("select count(id) as n from user", db.engine)
    nobs = pd.read_sql("select count(id) as n from data", db.engine)
    nsit = pd.read_sql("select count(id) as n from site", db.engine)
    return render_template('index.html',nobs="{:,}".format(nobs.n.sum()), nuse=nuse.n[0], nsit=nsit.n[0])

@app.route('/analytics')
def analytics():
    ss = pd.read_sql_table('site',db.engine).set_index(['site','region'])
    sqlq = 'select region, site, count(region) as value from data group by region, site'
    xx = pd.read_sql(sqlq, db.engine).set_index(['site','region'])
    res = xx.merge(ss,"left",left_index=True,right_index=True)
    res = res.reset_index()
    res = res[['region','site','name','latitude','longitude','value']]
    res = res.rename(columns={'region':'Region','site':'Site','name':'Name','latitude':'Latitude','longitude':'Longitude','value':'Observations'}).fillna(0).sort_values(['Observations','Longitude'],ascending=False)
    res.Observations = res.Observations.astype(int)
    return render_template('analytics.html',dats=Markup(res.to_html(index=False,classes=['table','table-condensed'])))

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':  # checks
        replace = False if request.form.get('replace') is None else True
        if 'file' not in request.files:
            flash('No file part','alert-danger')
            return redirect(request.url)
        ufiles = request.files.getlist("file")
        ufnms = [x.filename for x in ufiles]
        ld = os.listdir(app.config['UPLOAD_FOLDER'])
        ffregex = "[A-Z]{2}_.*_[0-9]{4}-[0-9]{2}-[0-9]{2}_[A-Z]{2}[0-9]?.[a-zA-Z]{3}" # core sites
        ffregex2 = "[A-Z]{2}_.*_[0-9]{4}-[0-9]{2}-[0-9]{2}.csv" # leveraged sites
        pattern = re.compile(ffregex+"|"+ffregex2)
        if not all([pattern.match(f) is not None for f in ufnms]):
            # file names do not match expected pattern
            flash('Please name your files with the specified format.','alert-danger')
            return redirect(request.url)
        if not replace: # not replacing files, need to check if files already exist
            if all([f in ld for f in ufnms]):
                # all files already uploaded
                flash('All of those files were already uploaded.','alert-danger')
                return redirect(request.url)
            if (any([f in ld for f in ufnms])):
                # remove files already uploaded
                ufiles = [f for f in ufiles if f not in ld]
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
                ver = len([x for x in ld if filename.split(".")[0] in x])
                # rename files if existing, add version number
                fup = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                if replace and ver>0: # need to add version number to file
                    fns = filename.split(".")
                    filename = fns[0]+"v"+str(ver+1)+"."+fns[1]
                    fup = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(fup) # save it as fup
                # sb.upload_file_to_item(sbupf, fup)
                filenames.append(filename) # this is the filename list that gets passed on for display
                fnlong.append(fup) # this is the list of files that get processed
        # PROCESS the data and save as tmp file
        try:
            if site[0] in core.index.tolist():
                gmtoff = core.loc[site].GMTOFF[0]
                x = sp_in(fnlong, gmtoff)
            else:
                if len(fnlong)>1:
                    flash('Please merge your data prior to upload.','alert-danger')
                    [os.remove(f) for f in fnlong]
                    return redirect(request.url)
                x = sp_in_lev(fnlong[0])
            tmp_file = site[0].encode('ascii')+"_"+binascii.hexlify(os.urandom(6))
            out_file = os.path.join(app.config['UPLOAD_FOLDER'],tmp_file+".csv")
            x.to_csv(out_file,index=False)
            columns = x.columns.tolist()
            rr,ss = site[0].split("_")
            cdict = pd.read_sql("select * from cols where region='"+rr+"' and site='"+ss+"'", db.engine)
            cdict = dict(zip(cdict['rawcol'],cdict['dbcol']))
            flash("Please double check your column matching. Thanks!",'alert-warning')
        except IOError:
            msg = Markup('Unknown error. Please <a href="mailto:aaron.berdanier@gmail.com" class="alert-link">email Aaron</a> with a copy of the file you tried to upload...')
            flash(msg,'alert-danger')
            [os.remove(f) for f in fnlong]
            return redirect(request.url)
        # check if existing site
        allsites = pd.read_sql("select concat(region,'_',site) as sitenm from site",db.engine).sitenm.tolist()
        existing = True if site[0] in allsites else False
        return render_template('upload_columns.html', filenames=filenames, columns=columns, tmpfile=tmp_file, variables=variables, cdict=cdict, existing=existing, sitenm=site[0], replacing=replace)
    if request.method == 'GET':
        xx = pd.read_sql("select distinct region, site from data", db.engine)
        vv = pd.read_sql("select distinct variable from data", db.engine)['variable'].tolist()
        sites = [x[0]+"_"+x[1] for x in zip(xx.region,xx.site)]
        sitedict = sorted([getsitenames(x) for x in sites], key=lambda tup: tup[1])
        return render_template('upload.html', sites=sitedict, variables = map(str,vv))

@app.route("/upload_cancel",methods=["POST"])
def cancelcolumns():
    ofiles = request.form['ofiles'].split(",")
    tmpfile = request.form['tmpfile']+".csv"
    ofiles.append(tmpfile)
    [os.remove(os.path.join(app.config['UPLOAD_FOLDER'],x)) for x in ofiles] # remove tmp files
    flash('Upload cancelled.','alert-primary')
    return redirect(url_for('upload'))

@app.route("/_addmanualdata",methods=["POST"])
def manual_upload():
    rgn, ste = request.json['site'].split("_")
    data = [d for d in request.json['data'] if None not in d] # get all complete rows
    dd = pd.DataFrame(data,columns=["DateTime_UTC","variable","value"])
    dd['DateTime_UTC'] = pd.to_datetime(dd['DateTime_UTC'],format='%Y-%m-%d %H:%M')
    dd['DateTime_UTC'] = pd.DatetimeIndex(dd['DateTime_UTC']).round("15Min")
    dd['region'] = rgn
    dd['site'] = ste
    dd['value'] = pd.to_numeric(dd['value'], errors='coerce')
    # region site datetime variable value
    dd = dd[['region','site','DateTime_UTC','variable','value']]
    dd.to_sql("manual", db.engine, if_exists='append', index=False, chunksize=1000)
    return jsonify(result="success")

@app.route("/policy")
def datapolicy():
    return render_template("datapolicy.html")

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

def updatedb(xx, replace=False):
    if replace: # need to go row by row... ugh!
        for r in xx.values:
            try: # if it exists, will return something and update the last one (since we grab the last one in the downloads)
                # don't know if the order_by has any performance cost (prob does)... may be a better way to do this.
                d = Data.query.order_by(Data.id.desc()).filter(Data.region==r[0], Data.site==r[1], Data.DateTime_UTC==r[2], Data.variable==r[3]).first_or_404()
                d.value = r[4]
            except: # doesn't exist, need to add it
                d = Data(r[0], r[1], r[2], r[3], r[4], r[5])
                db.session.add(d)
            db.session.commit()
    else:
        xx.to_sql("data", db.engine, if_exists='append', index=False, chunksize=1000)

@app.route("/upload_confirm",methods=["POST"]) # confirm columns
def confirmcolumns():
    cdict = json.loads(request.form['cdict'])
    tmpfile = request.form['tmpfile']
    cdict = dict([(r['name'],r['value']) for r in cdict])
    try: #something successful
        xx = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'],tmpfile+".csv"), parse_dates=[0])
        xx = xx[cdict.keys()].rename(columns=cdict)
        region, site = tmpfile.split("_")[:-1]
        # Add site if new site
        if request.form['existing'] == "no":
            # need to add site to list
            embargo = False #if request.form['policy']=="streampulse" else True
            usgss = None if request.form['usgs']=="" else request.form['usgs']
            sx = Site(region=region, site=site, name=request.form['sitename'],
                latitude=request.form['lat'], longitude=request.form['lng'], usgs=usgss,
                addDate=datetime.utcnow(), embargo=embargo, by=current_user.get_id(),
                contact=request.form['contactName'], contactEmail=request.form['contactEmail'])
            db.session.add(sx)
            db.session.commit()
            # need to make a new text file with the metadata
            metastring = request.form['metadata']
            metafilepath = os.path.join(app.config['META_FOLDER'],region+"_"+site+"_metadata.txt")
            with open(metafilepath, 'a') as metafile:
                metafile.write(metastring)
        xx = xx.set_index("DateTime_UTC")
        xx.columns.name = 'variable'
        xx = xx.stack()
        xx.name="value"
        xx = xx.reset_index()
        xx = xx.groupby(['DateTime_UTC','variable']).mean().reset_index() # average duplicates
        xx['region'] = region
        xx['site'] = site
        xx['flag'] = None
        xx = xx[['region','site','DateTime_UTC','variable','value','flag']]
        # add a check for duplicates?
        replace = True if request.form['replacing']=='yes' else False
        # xx.to_sql("data", db.engine, if_exists='append', index=False, chunksize=1000)
        updatedb(xx, replace)
        updatecdict(region, site, cdict)
    except IOError:
        flash('There was an error, please try again.','alert-warning')
        return redirect(request.url)
    os.remove(os.path.join(app.config['UPLOAD_FOLDER'],tmpfile+".csv")) # remove tmp file
    flash('Uploaded '+str(len(xx.index))+' values, thank you!','alert-success')
    return redirect(url_for('upload'))

def getsitenames(regionsite):
    region, site = regionsite.split("_")
    names = pd.read_sql("select name from site where region='"+region+"' and site ='"+site+"'", db.engine)
    return (regionsite, region+" - "+names.name[0])

@app.route('/download')
def download():
    vv = pd.read_sql("select distinct region, site, variable from data", db.engine)
    sites = [x[0]+"_"+x[1] for x in zip(vv.region,vv.site)]
    if current_user.is_authenticated:
        sites = authenticate_sites(sites, user=current_user.get_id())
    else:
        sites = authenticate_sites(sites)
    ss = []
    for site in sites:
        r,s = site.split("_")
        ss.append("(region='"+r+"' and site='"+s+"') ")
    qs = "or ".join(ss)
    nn = pd.read_sql("select region, site, name from site",db.engine)
    dd = pd.read_sql("select region, site, min(DateTime_UTC) as startdate, max(DateTime_UTC) as enddate from data where "+qs+"group by region, site", db.engine)
    vv = vv.groupby(['region','site'])['variable'].unique().reset_index()
    dx = pd.merge(vv, nn.merge(dd, on=['region','site'], how='right'), on=['region','site'], how='right')
    dx['regionsite'] = [x[0]+"_"+x[1] for x in zip(dx.region,dx.site)]
    dx.startdate = dx.startdate.apply(lambda x: x.strftime('%Y-%m-%d'))
    dx.enddate = dx.enddate.apply(lambda x: x.strftime('%Y-%m-%d'))
    dx.name = dx.region+" - "+dx.name
    dvv = dx[['regionsite','name','startdate','enddate','variable']].values
    sitedict = sorted([tuple(x) for x in dvv], key=lambda tup: tup[1])
    return render_template('download.html',sites=sitedict)

# CODE FROM OLD Download, depreciated
# xx = pd.read_sql("select distinct region, site from data", db.engine)
# sites = [x[0]+"_"+x[1] for x in zip(xx.region,xx.site)]
# # xx = pd.read_sql("select distinct concat(region,'_',site) as sites from data", db.engine)
# # sites = xx['sites'].tolist()
# # check login status... allow download without login for certain sites.
# if current_user.is_authenticated:
#     sites = authenticate_sites(sites, user=current_user.get_id())
# else:
#     sites = authenticate_sites(sites)
# # sites = [(x.split("_")[0],x) for x in xx.sites.tolist()]
# # sitedict = {'0ALL':'ALL'}
# # for x in sites:
# #     if x[0] not in sitedict:
# #         sitedict[x[0]] = []
# #     sitedict[x[0]].append(x[1])
# sitedict = sorted([getsitenames(x) for x in sites], key=lambda tup: tup[1])
# return render_template('download.html',sites=sitedict)

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
    sitenm = request.form['sites'].split(",")
    startDate = request.form['startDate']#.split("T")[0]
    endDate = request.form['endDate']
    variables = request.form['variables'].split(",")
    email = request.form.get('email')
    if current_user.get_id() is None: # not logged in
        uid = None
        if email is not None: # send email to aaron, add to csv
            elist = open("static/email_list.csv","a")
            elist.write(email+"\n")
            elist.close()
    else: # get logged in email
        uid = int(current_user.get_id())
        myuser = User.query.filter(User.id==uid).first()
        email = myuser.email
    ## add download stats
    dnld_stat = Downloads(timestamp=datetime.utcnow(), userID=uid, email=email,
        dnld_sites=request.form['sites'], dnld_date0=startDate,
        dnld_date1=endDate, dnld_vars=request.form['variables'])
    db.session.add(dnld_stat)
    db.session.commit()
    ## get data
    aggregate = request.form['aggregate']
    dataform = request.form['dataform'] # wide or long
    tmp = tempfile.mkdtemp()
    # add the data policy to the folder
    shutil.copy2("static/streampulse_data_policy.txt", tmp)
    # loop through sites
    for s in sitenm:
        # get data for site s
        sqlq = "select * from data where concat(region,'_',site)='"+s+"' "+\
            "and DateTime_UTC>'"+startDate+"' "+\
            "and DateTime_UTC<'"+endDate+"' "+\
            "and variable in ('"+"', '".join(variables)+"')"
        xx = pd.read_sql(sqlq, db.engine)
        if len(xx)<1:
            continue
        xx.loc[xx.flag==0,"value"] = None # set NA values
        xx.dropna(subset=['value'], inplace=True) # remove rows with NA value
        if request.form.get('flag') is not None:
            xx.drop(['id'], axis=1, inplace=True) # keep the flags
        else:
            xx.drop(['id','flag'], axis=1, inplace=True) # get rid of them
        if request.form.get('usgs') is not None:
            #xu = get_usgs([s],startDate,endDate)
            print xx['DateTime_UTC'].min()
            xu = get_usgs([s], xx['DateTime_UTC'].min().strftime("%Y-%m-%d"), xx['DateTime_UTC'].max().strftime("%Y-%m-%d"))
            if len(xu) is not 0:
                xx = pd.concat([xx,xu])
        # check for doubles with same datetime, region, site, variable...
        xx = xx.set_index(["DateTime_UTC","region","site","variable"])
        xx = xx[~xx.index.duplicated(keep='last')].sort_index().reset_index()
        if aggregate!="none":
            xx = xx.set_index(['DateTime_UTC']).groupby(['region','site','variable']).resample(aggregate).mean().reset_index()
        if dataform=="wide":
            xx = xx.pivot_table("value",['region','site','DateTime_UTC'],'variable').reset_index()
        xx.to_csv(tmp+'/'+s+'_data.csv', index=False)
        # copy metadata, if it exists
        mdfile = os.path.join(app.config['META_FOLDER'],s+"_metadata.txt")
        if os.path.isfile(mdfile):
            shutil.copy2(mdfile, tmp)
    #
    writefiles = os.listdir(tmp) # list files in the tmp directory
    zipname = 'SPdata_'+datetime.now().strftime("%Y-%m-%d")+'.zip'
    with zipfile.ZipFile(tmp+'/'+zipname,'w') as zf:
        [zf.write(tmp+'/'+f,f) for f in writefiles]
    #flash('File sent: '+zipname, 'alert-success')
    return send_file(tmp+'/'+zipname, 'application/zip', as_attachment=True, attachment_filename=zipname)

@app.route('/visualize')
# @login_required
def visualize():
    vv = pd.read_sql("select distinct region, site, variable from data", db.engine)
    sites = [x[0]+"_"+x[1] for x in zip(vv.region,vv.site)]
    if current_user.is_authenticated:
        sites = authenticate_sites(sites, user=current_user.get_id())
    else:
        sites = authenticate_sites(sites)
    ss = []
    for site in sites:
        r,s = site.split("_")
        ss.append("(region='"+r+"' and site='"+s+"') ")
    qs = "or ".join(ss)
    nn = pd.read_sql("select region, site, name from site",db.engine)
    dd = pd.read_sql("select region, site, min(DateTime_UTC) as startdate, max(DateTime_UTC) as enddate from data where "+qs+"group by region, site", db.engine)
    vv = vv.groupby(['region','site'])['variable'].unique().reset_index()
    dx = pd.merge(vv, nn.merge(dd, on=['region','site'], how='right'), on=['region','site'], how='right')
    dx['regionsite'] = [x[0]+"_"+x[1] for x in zip(dx.region,dx.site)]
    dx.startdate = dx.startdate.apply(lambda x: x.strftime('%Y-%m-%d'))
    dx.enddate = dx.enddate.apply(lambda x: x.strftime('%Y-%m-%d'))
    dx.name = dx.region+" - "+dx.name
    dvv = dx[['regionsite','name','startdate','enddate','variable']].values
    sitedict = sorted([tuple(x) for x in dvv], key=lambda tup: tup[1])
    return render_template('visualize.html',sites=sitedict)

# OLD CODE FROM VISUALIZE, depriciated
# return "success"
#
# # xx = pd.read_sql("select distinct concat(region,'_',site) as sites from data", db.engine)
# # sites = xx['sites'].tolist()
# xx = pd.read_sql("select distinct region, site from data", db.engine)
# sites = [x[0]+"_"+x[1] for x in zip(xx.region,xx.site)]
# if current_user.is_authenticated:
#     sites = authenticate_sites(sites, user=current_user.get_id())
# else:
#     sites = authenticate_sites(sites)
# sitedict = sorted([getsitenames(x) for x in sites], key=lambda tup: tup[1])

@app.route('/_getviz',methods=["POST"])
def getviz():
    region, site = request.json['site'].split(",")[0].split("_")
    startDate = request.json['startDate']
    endDate = request.json['endDate']#.split("T")[0]
    variables = request.json['variables']
    # print region, site, startDate, endDate, variables
    sqlq = "select * from data where region='"+region+"' and site='"+site+"' "+\
        "and DateTime_UTC>'"+startDate+"' "+\
        "and DateTime_UTC<'"+endDate+"' "+\
        "and variable in ('"+"', '".join(variables)+"')"
    xx = pd.read_sql(sqlq, db.engine)
    xx.loc[xx.flag==0,"value"] = None # set NaNs
    flagdat = xx[['DateTime_UTC','variable','flag']].dropna().drop(['flag'],axis=1).to_json(orient='records',date_format='iso') # flag data
    xx = xx.drop('id', axis=1).drop_duplicates()\
      .set_index(["DateTime_UTC","variable"])\
      .drop(['region','site','flag'],axis=1)
    xx = xx[~xx.index.duplicated(keep='last')].unstack('variable') # get rid of duplicated date/variable combos
    xx.columns = xx.columns.droplevel()
    xx = xx.reset_index()
    # Get sunrise sunset data
    sxx = pd.read_sql("select * from site where region='"+region+"' and site='"+site+"'",db.engine)
    sdt = datetime.strptime(startDate,"%Y-%m-%d")
    edt = datetime.strptime(endDate,"%Y-%m-%d")
    ddt = edt-sdt
    lat = sxx.latitude[0]
    lng = sxx.longitude[0]
    rss = []
    for i in range(ddt.days + 1):
        rise, sets = list(suns(sdt+timedelta(days=i-1), latitude=lat, longitude=lng).calculate())
        if rise>sets:
            sets = sets + timedelta(days=1) # account for UTC
        rss.append([rise, sets])
    #
    rss = pd.DataFrame(rss, columns=("rise","set"))
    rss.set = rss.set.shift(1)
    sunriseset = rss.loc[1:].to_json(orient='records',date_format='iso')
    return jsonify(variables=variables, dat=xx.to_json(orient='records',date_format='iso'), sunriseset=sunriseset, flagdat=flagdat)

@app.route('/clean')
@login_required
def qaqc():
    xx = pd.read_sql("select distinct region, site from data", db.engine)
    sitesa = [x[0]+"_"+x[1] for x in zip(xx.region,xx.site)]
    qaqcuser = current_user.qaqc_auth()
    sites = [z for z in sitesa if z in qaqcuser]
    xx = pd.read_sql("select distinct flag from flag", db.engine)
    flags = xx['flag'].tolist()
    sitedict = sorted([getsitenames(x) for x in sites], key=lambda tup: tup[1])
    return render_template('qaqc.html',sites=sitedict,flags=flags, tags=[''])

@app.route('/_getqaqc',methods=["POST"])
def getqaqc():
    region, site = request.json['site'].split(",")[0].split("_")
    sqlq = "select * from data where region='"+region+"' and site='"+site+"'"
    xx = pd.read_sql(sqlq, db.engine)
    xx.loc[xx.flag==0,"value"] = None # set NaNs
    flagdat = xx[['DateTime_UTC','variable','flag']].dropna().drop(['flag'],axis=1).to_json(orient='records',date_format='iso') # flag data
    #xx.dropna(subset=['value'], inplace=True) # remove rows with NA value
    variables = list(set(xx['variable'].tolist()))
    xx = xx.drop('id', axis=1).drop_duplicates()\
      .set_index(["DateTime_UTC","variable"])\
      .drop(['region','site','flag'],axis=1)
    xx = xx[~xx.index.duplicated(keep='last')].unstack('variable') # get rid of duplicated date/variable combos
    xx.columns = xx.columns.droplevel()
    xx = xx.reset_index()
    # Get sunrise sunset data
    sxx = pd.read_sql("select * from site where region='"+region+"' and site='"+site+"'",db.engine)
    sdt = min(xx.DateTime_UTC).replace(hour=0, minute=0,second=0,microsecond=0)
    edt = max(xx.DateTime_UTC).replace(hour=0, minute=0,second=0,microsecond=0)+timedelta(days=1)
    ddt = edt-sdt
    lat = sxx.latitude[0]
    lng = sxx.longitude[0]
    rss = []
    for i in range(ddt.days + 1):
        rise, sets = list(suns(sdt+timedelta(days=i-1), latitude=lat, longitude=lng).calculate())
        if rise>sets:
            sets = sets + timedelta(days=1) # account for UTC
        rss.append([rise, sets])
    #
    rss = pd.DataFrame(rss, columns=("rise","set"))
    rss.set = rss.set.shift(1)
    sunriseset = rss.loc[1:].to_json(orient='records',date_format='iso')
    # Get 2wk plot intervals
    def daterange(start, end):
        r = (end+timedelta(days=1)-start).days
        if r%14 > 0:
            r = r+14
        return [(end-timedelta(days=i)).strftime('%Y-%m-%d') for i in range(0,r,28)]
    #drr = pd.date_range(sdt,edt,freq="14D").tolist()[::-1] # need to reverse
    #drr = [da.strftime('%Y-%m-%d') for da in drr]
    drr = daterange(sdt,edt)
    return jsonify(variables=variables, dat=xx.to_json(orient='records',date_format='iso'), sunriseset=sunriseset, flagdat=flagdat, plotdates=drr)

@app.route('/_addflag',methods=["POST"])
def addflag():
    rgn, ste = request.json['site'].split("_")
    # sdt = dtparse.parse(request.json['startDate'])
    # edt = dtparse.parse(request.json['endDate'])
    sdt = datetime.strptime(request.json['startDate'],"%Y-%m-%dT%H:%M:%S.%fZ")
    edt = datetime.strptime(request.json['endDate'],"%Y-%m-%dT%H:%M:%S.%fZ")
    var = request.json['var']
    flg = request.json['flagid']
    cmt = request.json['comment']
    print request.json
    for vv in var:
        fff = Flag(rgn, ste, sdt, edt, vv, flg, cmt, int(current_user.get_id()))
        print fff
        db.session.add(fff)
        db.session.commit()
        flgdat = Data.query.filter(Data.region==rgn, Data.site==ste, Data.DateTime_UTC>=sdt, Data.DateTime_UTC<=edt, Data.variable==vv).all()
        print flgdat
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
    for vv in var:
        ttt = Tag(rgn, ste, sdt, edt, vv, tag, cmt, int(current_user.get_id()))
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
    sqlq = "select * from data where region='"+rgn+"' and site='"+ste+"'"
    xx = pd.read_sql(sqlq, db.engine)
    xx.loc[xx.flag==0,"value"] = None # set NaNs
    xx.dropna(subset=['value'], inplace=True) # remove rows with NA value
    xx = xx.drop('id', axis=1).drop_duplicates()\
      .set_index(["DateTime_UTC","variable"])\
      .drop(['region','site','flag'],axis=1)
    xx = xx[~xx.index.duplicated(keep='last')].unstack('variable') # get rid of duplicated date/variable combos
    xx.columns = xx.columns.droplevel()
    xx = xx.reset_index()
    return jsonify(dat=xx.to_json(orient='records',date_format='iso'))

@app.route('/cleandemo')
def qaqcdemo():
    sqlq = "select * from data where region='NC' and site='NHC' and "+\
        "DateTime_UTC>'2016-09-23' and DateTime_UTC<'2016-10-07' and "+\
        "variable in ('DO_mgL','WaterPres_kPa','CDOM_mV','Turbidity_mV','WaterTemp_C','pH','SpecCond_mScm')"
    # sqlq = "select * from data where region='NC' and site='Mud'"
    xx = pd.read_sql(sqlq, db.engine) # training data
    # xx.loc[xx.flag==0,"value"] = None # set NaNs existing flags
    flagdat = xx[['DateTime_UTC','variable','flag']].dropna().drop(['flag'],axis=1).to_json(orient='records',date_format='iso') # flag data
    variables = list(set(xx['variable'].tolist()))
    xx = xx.drop('id', axis=1).drop_duplicates()\
      .set_index(["DateTime_UTC","variable"])\
      .drop(['region','site','flag'],axis=1)\
      .unstack('variable')
    xx.columns = xx.columns.droplevel()
    xx = xx.reset_index()
        # get anomaly dates
    xtrain = xx[(xx.DateTime_UTC<'2016-09-29')].dropna()# training data first portion
    clf = svm.OneClassSVM(nu=0.01,kernel='rbf',gamma='auto')
    xsvm = xtrain.as_matrix(variables)
    clf.fit(xsvm)
    # xss.assign(pred=clf.predict(xsvm))
    xss = xx.dropna()
    xpred = xss.as_matrix(variables)
    xss['pred'] = clf.predict(xpred).tolist()
    anomaly = xss[xss.pred==-1].DateTime_UTC.to_json(orient='records',date_format='iso')
    # Get sunrise sunset data
    sxx = pd.read_sql("select * from site where region='NC' and site='NHC'",db.engine)
    sdt = min(xx.DateTime_UTC).replace(hour=0, minute=0,second=0,microsecond=0)
    edt = max(xx.DateTime_UTC).replace(hour=0, minute=0,second=0,microsecond=0)+timedelta(days=1)
    ddt = edt-sdt
    lat = sxx.latitude[0]
    lng = sxx.longitude[0]
    rss = []
    for i in range(ddt.days + 1):
        rise, sets = list(suns(sdt+timedelta(days=i-1), latitude=lat, longitude=lng).calculate())
        if rise>sets:
            sets = sets + timedelta(days=1) # account for UTC
        rss.append([rise, sets])
    #
    rss = pd.DataFrame(rss, columns=("rise","set"))
    rss.set = rss.set.shift(1)
    sunriseset = rss.loc[1:].to_json(orient='records',date_format='iso')
    return render_template('qaqcdemo.html', variables=variables, dat=xx.to_json(orient='records',date_format='iso'), sunriseset=sunriseset, flagdat=flagdat, anomaly=anomaly)

@app.route('/api')
def api():
    startDate = request.args.get('startdate')
    endDate = request.args.get('enddate')
    variables = request.args.get('variables')
    sites = request.args['sitecode'].split(',')
    if request.headers.get('Token') is not None:
        sites = authenticate_sites(sites, token=request.headers['Token'])
    elif current_user.is_authenticated:
        sites = authenticate_sites(sites, user=current_user.get_id())
    else:
        sites = authenticate_sites(sites)
    ss = []
    for site in sites:
        r,s = site.split("_")
        ss.append("(region='"+r+"' and site='"+s+"') ")
    qs = "or ".join(ss)
    # META
    meta = pd.read_sql("select region, site, name, latitude as lat, longitude as lon, usgs as usgsid from site where "+qs, db.engine)
    # DATA
    sqlq = "select * from data where "+qs
    if startDate is not None:
        sqlq = sqlq+"and DateTime_UTC>'"+startDate+"' "
    if endDate is not None:
        sqlq = sqlq+"and DateTime_UTC<'"+endDate+"' "
    if variables is not None:
        vvv = variables.split(",")
        sqlq = sqlq+"and variable in ('"+"', '".join(vvv)+"')"
    xx = pd.read_sql(sqlq, db.engine)
    vv = xx.variable.unique().tolist()
    xx.loc[xx.flag==0,"value"] = None # set NA values
    xx.dropna(subset=['value'], inplace=True) # remove rows with NA value
    # check if we need to get USGS data - if depth requested but not available
    xu = []
    if variables is not None:
        if "Discharge_m3s" in variables and "Discharge_m3s" not in vv:
            xu = get_usgs(sites, min(xx.DateTime_UTC).strftime("%Y-%m-%d"), max(xx.DateTime_UTC).strftime("%Y-%m-%d"))
        if "Depth_m" in variables and "Depth_m" not in vv and len(xu) is 0:
            xu = get_usgs(sites, min(xx.DateTime_UTC).strftime("%Y-%m-%d"), max(xx.DateTime_UTC).strftime("%Y-%m-%d"))
    if len(xu) is not 0:
        # subset usgs data based on each sites' dates...
        xx = pd.concat([xx,xu])
    # check for doubles with same datetime, region, site, variable...
    xx = xx.set_index(["DateTime_UTC","region","site","variable"])
    xx = xx[~xx.index.duplicated(keep='last')].sort_index().reset_index()
    xx['DateTime_UTC'] = xx['DateTime_UTC'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
    # FLAGS
    if request.args.get('flags')=='true':
        fsql = "select * from flag where "+qs
        if startDate is not None:
            fsql = fsql+"and startDate>'"+startDate+"' "
        if endDate is not None:
            fsql = fsql+"and endDate<'"+endDate+"' "
        if variables is not None:
            vvv = variables.split(",")
            fsql = fsql+"and variable in ('"+"', '".join(vvv)+"')"
        flags = pd.read_sql(fsql, db.engine)
        flags.drop(['by'], axis=1, inplace=True)
        xx.drop(['id'], axis=1, inplace=True)
        resp = jsonify(data=xx.to_dict(orient='records'),sites=meta.to_dict(orient='records'),flags=flags.to_dict(orient='records'))
    else:
        xx.drop(['id','flag'], axis=1, inplace=True)
        resp = jsonify(data=xx.to_dict(orient='records'),sites=meta.to_dict(orient='records'))
    return resp

@app.route('/model')
def modelgen():
    xx = pd.read_sql("select distinct region, site from data", db.engine)
    sites = [x[0]+"_"+x[1] for x in zip(xx.region,xx.site)]
    if current_user.is_authenticated:
        sites = authenticate_sites(sites, user=current_user.get_id())
    else:
        sites = authenticate_sites(sites)
    ss = []
    for site in sites:
        r,s = site.split("_")
        ss.append("(region='"+r+"' and site='"+s+"') ")
    qs = "or ".join(ss)
    nn = pd.read_sql("select region, site, name from site",db.engine)
    dd = pd.read_sql("select region, site, min(DateTime_UTC) as startdate, max(DateTime_UTC) as enddate from data where "+qs+"group by region, site", db.engine)
    # dd = pd.read_sql_table('data',db.engine)[['region','site','DateTime_UTC']]
    # dd = pd.concat([dd.groupby(['region','site']).DateTime_UTC.min(),dd.groupby(['region','site']).DateTime_UTC.max()], axis=1)
    # dd.columns = ['startdate','enddate']
    # dd = dd.reset_index()
    dx = nn.merge(dd, on=['region','site'], how='right')
    dx['regionsite'] = [x[0]+"_"+x[1] for x in zip(dx.region,dx.site)]
    dx.startdate = dx.startdate.apply(lambda x: x.strftime('%Y-%m-%d'))
    dx.enddate = dx.enddate.apply(lambda x: x.strftime('%Y-%m-%d'))
    dx.name = dx.region+" - "+dx.name
    sitedict = sorted([tuple(x) for x in dx[['regionsite','name','startdate','enddate']].values], key=lambda tup: tup[1])
    return render_template('model.html',sites=sitedict)

if __name__=='__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
