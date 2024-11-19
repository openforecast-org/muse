rm(list = ls())
source("R/ctll.R")
source("R/PTSfunctions.R")
source("R/PTSS3functions.R")
source("R/MSOEfunctions.R")
Rcpp::sourceCpp("src/musecpp2R.cpp")

x = 1000 + rnorm(1000, 0, 1)
ctll(x, silent=FALSE, log=T)

stop()

x = 1000 + rnorm(1000, 0, 1)
test = ctll(x, silent=FALSE, log=T)

stop()





# Intermittent example
y = ts(read.table("y.txt", header = FALSE, sep = "", dec = "."))
# Continuous time
h = 12
m = ctll(y, silent=F, h=h, log = TRUE, type="flow")



# plot
# plot(y[ind], type="l")
# lines(exp(m$comp[ind, 3]))
# lines(exp(m$comp[ind, 3] + 2 * sqrt(m$compV[ind, 3])))
aux = (length(y) + 1) : (length(y) + h)
print(cbind(exp(m$yFor), exp(m$yFor + 2 * sqrt(m$yForV))))
# plot with all data
# plot(y, type="l")
# lines(m$comp[, 3])


# PTS example
m = PTS(AirPassengers, model="zzz", verbose=TRUE)
m = PTSestim(m)
m = PTSsmooth(m)
m = PTSvalidate(m)
