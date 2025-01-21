rm(list = ls())
source("R/ctll.R")
source("R/PTSfunctions.R")
source("R/PTSS3functions.R")
source("R/MSOEfunctions.R")
Rcpp::sourceCpp("src/musecpp2R.cpp")

set.seed(41)
x <- c(rnorm(25,100,10),rnorm(25,110,10),rnorm(25,120,10),rnorm(25,150,10))
m = ctll(x, holdout=TRUE, h=10, log=TRUE)
fm = forecast(m, h=10)
plot(fm, legend=FALSE)
stop()

x = 1000 + rnorm(1000, 0, 1)
ctll(x, silent=FALSE, log=T)

stop()

x = 1000 + rnorm(1000, 0, 1)
test = ctll(x, silent=FALSE, log=T)

stop()

set.seed(41)
x <- rpois(100,1)
# stock y cllik=true
y = c(2.1648,2.1648,1.5621,1.7607,1.9846,2.2199,1.8373,
      1.3969,1.4835,1.1687,1.0759,1.0657,1.1250,1.3362,
      1.5869,1.8847,2.1093,2.4896,2.9386,2.5481,2.2095,
      1.9159,1.5914,1.3219,1.4515,1.1566,1.0661,1.0476,
      1.0293,1.0411,1.0964,1.2659,1.8502,2.5017,2.1156,
      1.8924,1.6018,2.1716,2.9441,2.9378,2.8708,2.8053,
      2.5634,1.5011,1.3195,1.1599,1.1826,1.4258,1.2255,
      1.2908,1.7550,1.3960,1.5500,1.7211,1.6446,1.5715,
      1.5016,1.4349,1.3711,1.3101,1.2519,1.4975,1.3413,
      1.2014,1.0761,1.0372,1.0369,1.0749,1.1978,1.3348,
      1.4874,1.6574,1.8469,1.9005,1.9557,1.9680,1.9486,
      1.8797,1.8133,1.7492,1.6874,1.6278,1.2781,1.2826,
      1.2871,1.6623,2.1470,1.9845,1.8343,1.5550,1.3182,
      1.4730,1.2122,1.2094,1.2066,1.4524,1.7482,2.1044,
      1.7769,1.3330,1.3330,1.3330,1.3330,1.3330,1.3330,
      1.3330,1.3330,1.3330,1.3330,1.3330,1.3330,1.3330)
plot(x, type = "l"); lines(y, col="red")


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
