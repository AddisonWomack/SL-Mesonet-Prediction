# Mesonet Data Preprocessing

# library with handy subsetting functions
require(caTools)

# read in input file that I've turned into a csv
df = read.csv2("C:/projectInputFiles/mesonetData.csv", header=TRUE, sep=",", stringsAsFactors = FALSE)

# don't actually want these columns
df$RMAX = NULL
df$RNUM = NULL


# remove bad observations from consideration
df = df[as.numeric(as.character(df$RAIN)) > -900,]
# more than a tenth of an inch of rain -- classify as true
df[as.numeric(as.character(df$RAIN)) > 0.1,]$RAIN = as.numeric(1)
# otherwise classify as false
df[as.numeric(as.character(df$RAIN)) <= 0.1,]$RAIN = as.numeric(0)

years = unique(df$YEAR)

# used to store resulting data frame of target variables for each date
target.df = data.frame(Year = as.integer(),
                Month = as.integer(),
                Day = as.integer(),
                WillRain = as.integer())

# iterate over all years
for (i in 1:length(years)) {
  currentYear = as.numeric(years[i])
  relevantDataForThisYear = df[as.numeric(df$YEAR) == currentYear,]
  months = unique(relevantDataForThisYear$MONTH)
  
  # iterate over all months within this year
  for (j in 1:length(months)) {
    currentMonth = as.numeric(months[j])
    relevantDataForThisMonth = relevantDataForThisYear[as.numeric(relevantDataForThisYear$MONTH) == currentMonth,]
    days = unique(relevantDataForThisMonth$DAY)
    
    # iterate over all days within this month
    for (k in 1:length(days)) {
      currentDay = as.numeric(days[k])
      relevantDataForThisDay = relevantDataForThisMonth[as.numeric(relevantDataForThisMonth$DAY) == currentDay,]
      didItRain = 0
      # did it rain at the majority of the stations for this day?
      # since data has aready been classified as a 0 or a 1 for each observation, 0.5, represents 50% of all stations
      if (mean(as.numeric(relevantDataForThisDay$RAIN)) >= 0.5) {
        didItRain = 1
      }
      
      # append new row to target frame
      newRow = data.frame(currentYear, currentMonth, currentDay, didItRain)
      target.df = data.frame(rbind(as.matrix(target.df), as.matrix(newRow)))
    }
    
  }
  
}

# write testing/training to separate csv files
write.csv(target.df, "C:/projectInputFiles/didItRainToday.csv", row.names = FALSE)
