library(ncdf4)
library(data.table)
require(caTools)

# precipiation
apcp24.ncin = nc_open("C:/input/apcp_sfc_latlon_24.nc")
apcp36.ncin = nc_open("C:/input/apcp_sfc_latlon_36.nc")

names(apcp24.ncin$var)
time = ncvar_get(apcp24.ncin, varid= "intTime")
precip24 = ncvar_get(apcp24.ncin, varid = "Total_precipitation")
precip36 = ncvar_get(apcp36.ncin, varid = "Total_precipitation")

rm(apcp24.ncin)
rm(apcp36.ncin)

# convection inhibition
cin24.ncin = nc_open("C:/input/cin_sfc_latlon_24.nc")
cin36.ncin = nc_open("C:/input/cin_sfc_latlon_36.nc")

names(cin24.ncin$var)
cin24 = ncvar_get(cin24.ncin, varid = "Convective_inhibition")
cin36 = ncvar_get(cin36.ncin, varid = "Convective_inhibition")

rm(cin24.ncin)
rm(cin36.ncin)

# surface pressure
press24.ncin = nc_open("C:/input/pres_sfc_latlon_24.nc")
press36.ncin = nc_open("C:/input/pres_sfc_latlon_36.nc")

names(press24.ncin$var)
press24 = ncvar_get(press24.ncin, varid = "Pressure_surface")
press36 = ncvar_get(press36.ncin, varid = "Pressure_surface")

rm(press24.ncin)
rm(press36.ncin)

# precipitable water
pwat24.ncin = nc_open("C:/input/pwat_eatm_latlon_24.nc")
pwat36.ncin = nc_open("C:/input/pwat_eatm_latlon_36.nc")

names(pwat24.ncin$var)
pwat24 = ncvar_get(pwat24.ncin, varid = "Precipitable_water")
pwat36 = ncvar_get(pwat36.ncin, varid = "Precipitable_water")

rm(pwat24.ncin)
rm(pwat36.ncin)

# specific humidity
spfh24.ncin = nc_open("C:/input/spfh_2m_latlon_24.nc")
spfh36.ncin = nc_open("C:/input/spfh_2m_latlon_36.nc")

names(spfh24.ncin$var)
spfh24 = ncvar_get(spfh24.ncin, varid = "Specific_humidity_height_above_ground")
spfh36 = ncvar_get(spfh36.ncin, varid = "Specific_humidity_height_above_ground")

rm(spfh24.ncin)
rm(spfh36.ncin)

# total cloud cover
tcdc24.ncin = nc_open("C:/input/tcdc_eatm_latlon_24.nc")
tcdc36.ncin = nc_open("C:/input/tcdc_eatm_latlon_36.nc")

names(tcdc24.ncin$var)
tcdc24 = ncvar_get(tcdc24.ncin, varid = "Total_cloud_cover")
tcdc36 = ncvar_get(tcdc36.ncin, varid = "Total_cloud_cover")

rm(tcdc24.ncin)
rm(tcdc36.ncin)

# column integrated condensate
tcol24.ncin = nc_open("C:/input/tcolc_eatm_latlon_24.nc")
tcol36.ncin = nc_open("C:/input/tcolc_eatm_latlon_36.nc")

names(tcol24.ncin$var)
tcol24 = ncvar_get(tcol24.ncin, varid = "Total_Column-Integrated_Condensate")
tcol36 = ncvar_get(tcol36.ncin, varid = "Total_Column-Integrated_Condensate")

rm(tcol24.ncin)
rm(tcol36.ncin)

# temperature
tmp24.ncin = nc_open("C:/input/tmp_2m_latlon_24.nc")
tmp36.ncin = nc_open("C:/input/tmp_2m_latlon_36.nc")

names(tmp24.ncin$var)
tmp24 = ncvar_get(tmp24.ncin, varid = "Temperature_height_above_ground")
tmp36 = ncvar_get(tmp36.ncin, varid = "Temperature_height_above_ground")

rm(tmp24.ncin)
rm(tmp36.ncin)

# u-wind
ugrd24.ncin = nc_open("C:/input/ugrd_10m_latlon_24.nc")
ugrd36.ncin = nc_open("C:/input/ugrd_10m_latlon_36.nc")

names(ugrd24.ncin$var)
uwind24 = ncvar_get(ugrd24.ncin, varid = "U-component_of_wind_height_above_ground")
uwind36 = ncvar_get(ugrd36.ncin, varid = "U-component_of_wind_height_above_ground")

rm(ugrd24.ncin)
rm(ugrd36.ncin)

# v-wind
vgrd24.ncin = nc_open("C:/input/vgrd_10m_latlon_24.nc")
vgrd36.ncin = nc_open("C:/input/vgrd_10m_latlon_36.nc")

names(vgrd24.ncin$var)
vwind24 = ncvar_get(vgrd24.ncin, varid = "V-component_of_wind_height_above_ground")
vwind36 = ncvar_get(vgrd36.ncin, varid = "V-component_of_wind_height_above_ground")

rm(vgrd24.ncin)
rm(vgrd36.ncin)

dim1 = dim(vwind24)[1]
dim2 = dim(vwind24)[2]

# read in predictor variables
target.df = read.csv2("C:/input/didItRainToday.csv", sep = ",", header = TRUE, stringsAsFactors = FALSE)

numberOfRecords = dim1 * dim2 * length(time)

# used to store resulting data frame of predictor variables for each date for each square
#reforecast.df = data.frame(year = rep(NA,numberOfRecords),
#                           month = rep(NA,numberOfRecords),
#                           day = rep(NA,numberOfRecords),
 #                          precip = rep(NA,numberOfRecords),
#                           specificHumidity = rep(NA,numberOfRecords),
 #                          uwind = rep(NA,numberOfRecords),
 #                          vwind = rep(NA,numberOfRecords),
 #                          willRain = rep(NA,numberOfRecords))


dt <- data.table(month = rep(0, numberOfRecords),
                 cin24=rep(0,numberOfRecords), 
                 cin36=rep(0,numberOfRecords),
                 precip24=rep(0,numberOfRecords),
                 precip36=rep(0,numberOfRecords),
                 press24=rep(0,numberOfRecords),
                 press36=rep(0,numberOfRecords),
                 pwat24=rep(0,numberOfRecords),
                 pwat36=rep(0,numberOfRecords),
                 spfh24=rep(0,numberOfRecords),
                 spfh36=rep(0,numberOfRecords),
                 tcdc24=rep(0,numberOfRecords),
                 tcdc36=rep(0,numberOfRecords),
                 tcol24=rep(0,numberOfRecords),
                 tcol36=rep(0,numberOfRecords),
                 tmp24=rep(0,numberOfRecords),
                 tmp36=rep(0,numberOfRecords),
                 uwind24=rep(0,numberOfRecords),
                 uwind36=rep(0,numberOfRecords),
                 vwind24=rep(0,numberOfRecords),
                 vwind36=rep(0,numberOfRecords),
                 willRain = rep(0,numberOfRecords)
                 )

numberOfDays = length(time)

for(i in 1:numberOfDays) {
  
  timeAsString = as.character(time[i])
  
  currentYearAsString = substr(timeAsString, 0, 4)
  currentMonthAsString = substr(timeAsString, 5, 6)
  currentDayAsString = substr(timeAsString, 7, 8)
  
  dateAsString = paste(currentYearAsString, currentMonthAsString, currentDayAsString, sep = "-")
  
  tomorrow = as.Date(dateAsString) + 1
  
  tomorrowYear = as.numeric(format(tomorrow,"%Y"))
  tomorrowMonth = as.numeric(format(tomorrow,"%m"))
  tomorrowDay = as.numeric(format(tomorrow,"%d"))
  
  currentYear = as.numeric(currentYearAsString)
  currentMonth = as.numeric(currentMonthAsString)
  currentDay = as.numeric(currentDayAsString)
  
  print(paste(currentYearAsString, currentMonthAsString, currentDayAsString, sep = " "))
  
  will.Rain = target.df[target.df$Year == tomorrowYear & target.df$Month == tomorrowMonth & target.df$Day == tomorrowDay,]$WillRain[1]
  
  for (j in 1:dim1) {
    for (k in 1:dim2) {
      
      jFactor = (j - 1) * dim1
      dayFactor = (i - 1) * (dim1 * dim2)
      kFactor = k - (j - 1)
      
      index = jFactor + dayFactor + kFactor

      cin.24 = cin24[j,k,i]
      cin.36 = cin36[j,k,i]
      precip.24 = precip24[j,k,i]
      precip.36 = precip36[j,k,i]
      press.24 = press24[j,k,i]
      press.36 = press36[j,k,i]
      pwat.24 = pwat24[j,k,i]
      pwat.36 = pwat36[j,k,i]
      spfh.24 = spfh24[j,k,i]
      spfh.36 = spfh36[j,k,i]
      tcdc.24 = tcdc24[j,k,i]
      tcdc.36 = tcdc36[j,k,i]
      tcol.24 = tcol24[j,k,i]
      tcol.36 = tcol36[j,k,i]
      tmp.24 = tmp24[j,k,i]
      tmp.36 = tmp36[j,k,i]
      uwind.24 = uwind24[j,k,i]
      uwind.36 = uwind36[j,k,i]
      vwind.24 = vwind24[j,k,i]
      vwind.36 = vwind36[j,k,i]
      
      dt[index, month := currentMonth]
      dt[index, cin24 := cin.24]
      dt[index, cin36 := cin.36]
      dt[index, precip24 := precip.24]
      dt[index, precip36 := precip.36]
      dt[index, press24 := press.24]
      dt[index, press36 := press.36]
      dt[index, pwat24 := pwat.24]
      dt[index, pwat36 := pwat.36]
      dt[index, spfh24 := spfh.24]
      dt[index, spfh36 := spfh.36]
      dt[index, tcdc24 := tcdc.24]
      dt[index, tcdc36 := tcdc.36]
      dt[index, tcol24 := tcol.24]
      dt[index, tcol36 := tcol.36]
      dt[index, tmp24 := tmp.24]
      dt[index, tmp36 := tmp.36]
      dt[index, uwind24 := uwind.24]
      dt[index, uwind36 := uwind.36]
      dt[index, vwind24 := vwind.24]
      dt[index, vwind36 := vwind.36]
      dt[index, willRain := will.Rain]
      

    }
  }
}

# convert table to data frame
df = as.data.frame(dt)

# remove na entries
df = na.omit(df)

# 10% of data set is for testing
isTesting = sample.split(df$precip24, SplitRatio = 0.1)

df.nonTesting = subset(df, isTesting == FALSE)
df.testing = subset(df, isTesting == TRUE)

isValidation = sample.split(df.nonTesting$precip24, SplitRatio = 0.15)
df.validation = subset(df.nonTesting, isValidation == TRUE)
df.training = subset(df.nonTesting, isValidation == FALSE)

# Part D
# Normalize all columns
for(varName in names(df.nonTesting)) {
  if (varName != "willRain" & varName != "month" ) {
    currentColumn = as.numeric(unlist(df.nonTesting[varName]))
    averageVal = mean(currentColumn)
    standardDev = sd(currentColumn)
    df.training[varName] = ((as.numeric(unlist(df.training[varName]))) - averageVal) / standardDev
    df.validation[varName] = ((as.numeric(unlist(df.validation[varName]))) - averageVal) / standardDev
    df.testing[varName] = ((as.numeric(unlist(df.testing[varName]))) - averageVal) / standardDev
  }
}

# write testing/training to separate csv files
write.csv(df.testing, "C:/input/testingSet.csv", row.names = FALSE)
write.csv(df.training, "C:/input/trainingSet.csv", row.names = FALSE)
write.csv(df.validation, "C:/input/validationSet.csv", row.names = FALSE)
write.csv(df.nonTesting, "C:/input/nonTestingSet.csv", row.names = FALSE)
write.csv(df, "C:/input/fullSet.csv", row.names = FALSE)


