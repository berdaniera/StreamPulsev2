na_fill = function(x, tol){
  # x is a vector of data
  # tol is max number of steps missing (if greater, it retains NA)
  ina = is.na(x)
  csum = cumsum(!ina)
  wg = as.numeric(names(which(table(csum) > tol))) # which gaps are too long
  x[ina] = approx(x, xout=which(ina))$y
  x[which(csum%in%wg)[-1]] = NA
  return(x)
}

sp_data = function(sitecode, startdate=NULL, enddate=NULL, variables=NULL, flags=FALSE, token=NULL){
    # sitecode is a site name
    # startdate and enddate are YYYY-MM-DD strings, e.g., '1983-12-09'
    # variables is a vector of c('variable_one', ..., 'variable_n')
    # flags is logical, include flag data or not
    u = paste0("http://data.streampulse.org/api?sitecode=",sitecode)
    if(!is.null(startdate)) u = paste0(u,"&startdate=",startdate)
    if(!is.null(enddate)) u = paste0(u,"&enddate=",enddate)
    if(!is.null(variables)) u = paste0(u,"&variables=",paste0(variables, collapse=","))
    if(flags) u = paste0(u,"&flags=true")
    cat(paste0('URL: ',u,'\n'))
    if(is.null(token)){
        r = httr::GET(u)
    }else{
        r = httr::GET(u, httr::add_headers(Token = token))
    }
    json = httr::content(r, as="text", encoding="UTF-8")
    d = jsonlite::fromJSON(json)
    d$data$DateTime_UTC = as.POSIXct(d$data$DateTime_UTC,tz="UTC")
    return(d)
}

sp_prep_metab = function(d){
  dd = d$data
  # rename USGSDepth_m and USGSDischarge_m3s
  if("USGSDepth_m"%in%dd$variable && !"Depth_m"%in%dd$variable){
      dd$variable[dd$variable=="USGSDepth_m"] = "Depth_m"
  }
  if("USGSDischarge_m3s"%in%dd$variable && !"Discharge_m3s"%in%dd$variable){
      dd$variable[dd$variable=="USGSDischarge_m3s"] = "Discharge_m3s"
  }
  vd = unique(dd$variable)
  dd = tidyr::spread(dd, variable, value) # need to reshape...
  # check if sufficient data
  md = d$sites # metadata
  # force into 15 minute intervals
  alldates = data.frame(DateTime_UTC=seq(dd[1,1],dd[nrow(dd),1],by="15 min"))
  dd = full_join(alldates,dd, by="DateTime_UTC")
  # calculate/define model variables
  dd$solar.time = suppressWarnings(streamMetabolizer::convert_UTC_to_solartime(date.time=dd$DateTime_UTC, longitude=md$lon[1], time.type="mean solar"))
  if("Light_PAR"%in%vd){
      dd$light = dd$Light_PAR
  }else{
      apparentsolartime = suppressWarnings(streamMetabolizer::convert_UTC_to_solartime(date.time=dd$DateTime_UTC, longitude=md$lon[1], time.type="apparent solar"))
      cat("NOTE: Modeling PAR based on location and date.\n")
      dd$light = suppressWarnings(streamMetabolizer::calc_solar_insolation(app.solar.time=apparentsolartime, latitude=md$lat[1], format="degrees"))
  }
  if("DO_mgL"%in%vd) dd$DO.obs = dd$DO_mgL
  if("Depth_m"%in%vd) dd$depth = dd$Depth_m
  if("WaterTemp_C"%in%vd) dd$temp.water = dd$WaterTemp_C
  if("Discharge_m3s"%in%vd) dd$discharge = dd$Discharge_m3s
  if("satDO_mgL"%in%vd){
      dd$DO.sat = dd$satDO_mgL
  }else{
      if("DOsat_pct"%in%vd){
          # define multiplicative factor, to catch if variable is fraction or percent... do this on server first in future!
          if(quantile(dd$DOsat_pct,0.9,na.rm=T)>10){ ff=0.01 }else{ ff=1 }
          dd$DO.sat = dd$DO.obs/(dd$DOsat_pct*ff)
      }else{
          if(!all(c("temp.water","AirPres_kPa")%in%colnames(dd))){
            stop("Insufficient data to fit this model.")
          }
          cat("NOTE: Modeling DO.sat based on water temperature and air pressure.\n")
          dd$DO.sat = LakeMetabolizer::o2.at.sat.base(temp = dd$temp.water, baro = dd$AirPres_kPa*10, salinity = 0, model = 'garcia-benson')
      }
  }
  return(dd)
}

sp_data_metab = function(sitecode, startdate=NULL, enddate=NULL, type="bayes", token=NULL){
    # return data for streamMetabolizer metab()
    # sitecode is a site name
    # startdate and enddate are strings "2016-12-11"
    if(length(sitecode)>1) stop("Please only enter one site to model.")
    if(is.null(startdate)&is.null(enddate)){
        if(as.Date(enddate)<as.Date(startdate)) stop("Start date is after end date.")
    }
    # Add: check for type, decide on what variables to include
    variables = c("DO_mgL","satDO_mgL","DOsat_pct","Depth_m","WaterTemp_C","Light_PAR","AirPres_kPa","Discharge_m3s")
    #c("DO_mgL","satDO_mgL","Depth_m","WaterTemp_C","Light_PAR","AirPres_kPa")
    cat("Downloading data from StreamPULSE.\n")
    d = sp_data(sitecode, startdate, enddate, variables, FALSE, token)
    cat("Formatting data for streamMetabolizer.\n")
    dd = sp_prep_metab(d)
    model_variables = c("solar.time","DO.obs","DO.sat","depth","temp.water","light")
    if(type=="bayes") model_variables = c(model_variables,"discharge")
    if(!all(model_variables%in%colnames(dd))){
        stop("Insufficient data to fit this model.")
    }
    fitdata = dplyr::select_(dd, .dots=model_variables)
    # gap fill linearly if less than 3h
    fitdata = data.frame(solar.time=fitdata[,1],apply(fitdata[,2:ncol(fitdata)],2,na_fill,tol=12))
    return(fitdata)
}
