# read in input file that I've removed first row from and replaced two space separators with one space separators
df = read.csv2("C:/projectInputFiles/trainingSet.csv", header=TRUE, sep=",", stringsAsFactors = FALSE)
head(df)

df.rainy = df[df$willRain == 1,]
nrow(df.rainy)

# augment dataset with rainy days
df = data.frame(rbind(as.matrix(df), as.matrix(df.rainy)))
df = data.frame(rbind(as.matrix(df), as.matrix(df.rainy)))
df = data.frame(rbind(as.matrix(df), as.matrix(df.rainy)))
df = data.frame(rbind(as.matrix(df), as.matrix(df.rainy)))
df = data.frame(rbind(as.matrix(df), as.matrix(df.rainy)))
df = data.frame(rbind(as.matrix(df), as.matrix(df.rainy)))

# shuffle dataset
df = df[sample(nrow(df)),]


# write testing/training to separate csv files
write.csv(df, "C:/projectInputFiles/augmentedTrainingSet.csv", row.names = FALSE)