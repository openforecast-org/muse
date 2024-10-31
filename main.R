rm(list = ls())
source("R/INTLEVEL.R")
source("R/PTSfunctions.R")
source("R/PTSS3functions.R")
source("R/MSOEfunctions.R")
Rcpp::sourceCpp("src/INTLEVELcpp2R.cpp")
Rcpp::sourceCpp("src/MSOEc.cpp")

# PTS example
m = PTS(AirPassengers, model="zzz", verbose=TRUE)
m = PTSestim(m)
m = PTSsmooth(m)
m = PTSvalidate(m)
# Intermittent example
y = ts(read.table("y.txt", header = FALSE, sep = "", dec = "."))
# Continuous time
h = 12
m = funName(y, verbose = TRUE, h=h, logTransform = TRUE)
# plot
# plot(y[ind], type="l")
# lines(exp(m$comp[ind, 3]))
# lines(exp(m$comp[ind, 3] + 2 * sqrt(m$compV[ind, 3])))
aux = (length(y) + 1) : (length(y) + h)
print(cbind(exp(m$yFor), exp(m$yFor + 2 * sqrt(m$yForV))))
# plot with all data
# plot(y, type="l")
# lines(m$comp[, 3])
