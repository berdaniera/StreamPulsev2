# -*- coding: utf-8 -*-
from flask import Flask, Markup, session, flash, render_template, request, jsonify, url_for, make_response, redirect, g
from flask_login import LoginManager, login_user, logout_user, current_user, login_required
from werkzeug.security import generate_password_hash, check_password_hash
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
import pysb
import os
import re

app = Flask(__name__)
app.config['SECRET_KEY'] = "insecure"
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:pass@localhost/sp'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = '/home/aaron/spv2/static/uploads'

########## DATABASE
#sb = pysb.SbSession()
#sb.login("abb30@duke.edu","BerdanierXpbrec1!")
#sbupf = sb.get_item("58471447e4b0f34b016ff277")
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
    def __init__(self, region, site, DateTime_UTC, variable, value):
        self.region = region
        self.site = site
        self.DateTime_UTC = DateTime_UTC
        self.variable = variable
        self.value = value
    def __repr__(self):
        return '<Data %r, %r, %r, %r>' % (self.region, self.site, self.DateTime_UTC, self.variable)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(55), unique=True, index=True)
    password = db.Column(db.String(255))
    token = db.Column(db.String(100), nullable=False, server_default='')
    email = db.Column(db.String(255), unique=True)
    registered_on = db.Column(db.DateTime())
    confirmed = db.Column(db.Boolean)
    def __init__(self, username, password, email):
        self.username = username
        self.set_password(password)
        self.token = binascii.hexlify(os.urandom(10))
        self.email = email
        self.registered_on = datetime.utcnow()
        self.confirmed = True
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
core = pd.read_csv('/home/aaron/spv2/static/sitelist.csv')
core['SITECD'] = list(core["REGIONID"].map(str) +"_"+ core["SITEID"])
core = core.set_index('SITECD')

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
'WaterTemp_C',
'WaterPres_kPa',
'AirTemp_C',
'AirPres_kPa',
'Depth_m',
'Discharge_m3s',
'Discharge_cfs',
'Velocity_ms',
'pH',
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

def load_file(f, gmtoff, logger):
    if logger=="CS":
        return read_csci(f, gmtoff)
    elif "H" in logger:
        return read_hobo(f)
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
    x = x.set_index("DateTime_UTC").sort_index().resample('15Min').mean()
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
@login_required
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
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
                gmtoff = core.loc[site].GMTOFF
                x = sp_in(fnlong, gmtoff)
            else:
                x = sp_in_lev(fnlong)
            tmp_file = site[0].encode('ascii')+"_"+binascii.hexlify(os.urandom(6))
            out_file = os.path.join(app.config['UPLOAD_FOLDER'],tmp_file+".csv")
            x.to_csv(out_file,index=False)
            columns = x.columns.tolist()
        except IOError:
            flash('Unknown error. Please email Aaron...','alert-danger')
            [os.remove(f) for f in fnlong]
            return redirect(request.url)
        return render_template('upload_columns.html', filenames=filenames, columns=columns, tmpfile=tmp_file, variables=variables)
    return render_template('upload.html')

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
        xx = xx[['region','site','DateTime_UTC','variable','value']]
        # add a check for duplicates?
        xx.to_sql("data", db.engine, if_exists='append', index=False)
    except IOError:
        flash('There was an error, try again.','alert-warning')
        return redirect(request.url)
    os.remove(os.path.join(app.config['UPLOAD_FOLDER'],tmpfile+".csv")) # remove tmp file
    flash('Uploaded '+str(len(xx.index))+' values, thank you!','alert-success')
    return redirect(url_for('upload'))#jsonify(result="Success",nin=len(xx.index))

# l = xx.columns.tolist()
# dups = [x for x in l if l.count(x) > 1]
# dict([(dups[d], dups[d]+str(d)) if d>0 else (dups[d], dups[d]) for d in range(len(dups))])
# tmpfile = 'fe1caa080a6b57a85e81'
# cdict
# cdict['water_kPa'] = 'WaterTemp_C'
# x2 = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'],tmpfile+".csv"),parse_dates=[0])
# xx = x2[cdict.keys()].rename(columns=cdict)
# xx.head()
# #region, site = tmpfile.split("_")[:-1]
# xx = xx.set_index("DateTime_UTC")
# xx.columns.name = 'variable'
# xx = xx.stack()
# xx.name="value"
# xx = xx.reset_index()
# xx['region'] = region
# xx['site'] = site
# xx = xx[['region','site','DateTime_UTC','variable','value']]
# xx.head()

@app.route('/download')
def download():
    xx = pd.read_sql("select distinct concat(region,'_',site) as sites from data", db.engine)
    sites = xx['sites'].tolist()
    return render_template('download.html',sites=sites)

@app.route('/_getstats',methods=['POST'])
def getstats():
    sitenm = request.json['site']
    # sites = [s.split("_") for s in sitenm]
    xx = pd.read_sql("select * from data where concat(region,'_',site) in ('"+"', '".join(sitenm)+"')", db.engine)
    startDate = xx.DateTime_UTC.min().strftime("%Y-%m-%d")
    endDate = (xx.DateTime_UTC.max()+timedelta(days=1)).strftime("%Y-%m-%d")
    xx = pd.read_sql("select distinct variable from data", db.engine)
    variables = xx['variable'].tolist()
    return jsonify(result="Success", startDate=startDate, endDate=endDate, variables=variables, site=sitenm)

@app.route('/_getcsv',methods=["POST"])
def getcsv():
    sitenm = request.form['site'].split(",")
    startDate = request.form['startDate']#.split("T")[0]
    endDate = request.form['endDate']
    variables = request.form['variables'].split(",")
    sqlq = "select * from data where concat(region,'_',site) in ('"+"', '".join(sitenm)+"') and DateTime_UTC>'"+startDate+"' and DateTime_UTC<'"+endDate+"' and variable in ('"+"', '".join(variables)+"')"
    xx = pd.read_sql(sqlq, db.engine)
    xx.drop('id', axis=1, inplace=True)
    resp = make_response(xx.to_csv(index=False))
    resp.headers["Content-Disposition"] = "attachment; filename=export.csv"
    resp.headers["Content-Type"] = "text/csv"
    return resp

@app.route('/visualize')
def visualize():
    xx = pd.read_sql("select distinct concat(region,'_',site) as sites from data", db.engine)
    sites = xx['sites'].tolist()
    return render_template('visualize.html',sites=sites)

@app.route('/_getviz',methods=["POST"])
def getviz():
    sitenm = request.json['site'].split(",")
    startDate = request.json['startDate'].split("T")[0]
    endDate = request.json['endDate'].split("T")[0]
    variables = request.json['variables']
    print sitenm, startDate, endDate, variables
    print request.json['endDate']
    sqlq = "select * from data where concat(region,'_',site) in ('"+"', '".join(sitenm)+"') "+\
        "and DateTime_UTC>'"+startDate+"' "+\
        "and DateTime_UTC<'"+endDate+"' "+\
        "and variable in ('"+"', '".join(variables)+"')"
    print sqlq
    xx = pd.read_sql(sqlq, db.engine)
    xx = xx.drop('id', axis=1).drop_duplicates()\
      .set_index(["DateTime_UTC","variable"])\
      .drop(['region','site'],axis=1)\
      .unstack('variable')
    xx.columns = xx.columns.droplevel()
    xx = xx.reset_index()
    print xx.head()
    return jsonify(variables=variables, dat=xx.to_json(orient='records',date_format='iso'))


# # startDate = xx.DateTime_UTC.min().strftime("%Y-%m-%d")
# # endDate = (xx.DateTime_UTC.max()+timedelta(days=1)).strftime("%Y-%m-%d")
# # cur = db.connection.cursor()
# # cur.execute("select distinct variable from data")
# # variables = [row[0] for row in cur.fetchall()]
# xx.head()
#
# x2 = xx.to_json(orient='records',date_format="iso")
#
# region = "NC"
# site = "Eno"
#
# import MySQLdb
# dbx = MySQLdb.connect(host="localhost",user="root",passwd="pass",db="sp")
# cur = dbx.cursor()
# cur.execute("select distinct concat(region,'_',site) from data")
# [row[0] for row in cur.fetchall()]
# db.close()

@app.route('/qaqc')
def qaqc():
    xx = pd.read_sql("select distinct concat(region,'_',site) as sites from data", db.engine)
    sites = xx['sites'].tolist()
    flags = ['a','b']
    return render_template('qaqc.html',sites=sites,flags=flags)

@app.route('/_getqaqc',methods=["POST"])
def getqaqc():
    sitenm = request.json['site']
    xx = pd.read_sql("select distinct variable from data", db.engine)
    variables = xx['variable'].tolist()
    sqlq = "select * from data where concat(region,'_',site) = '"+sitenm+"'"
    xx = pd.read_sql(sqlq, db.engine)
    xx = xx.drop('id', axis=1).drop_duplicates()\
      .set_index(["DateTime_UTC","variable"])\
      .drop(['region','site'],axis=1)\
      .unstack('variable')
    xx.columns = xx.columns.droplevel()
    xx = xx.reset_index()
    print variables
    return jsonify(variables=variables, dat=xx.to_json(orient='records',date_format='iso'))


@app.route('/visualize')
def view():
    return "x"


if __name__=='__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
