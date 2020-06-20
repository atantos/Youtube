# Loading the necessary packages
# Package for fitting regression models
using GLM

# Package for manipulating data frames
using DataFrames

# Package for importing/exporting csv files into/from data frames
using CSV

# Reading a csv file from github into a data frame stored to the variable GLC_test4
GLC_test4 = CSV.read(download("https://raw.githubusercontent.com/atantos/Youtube/master/GLC_test4.csv"),
                     decimal=',')

# Reading a local csv file
GLC_test4 = CSV.read("<path_to_file>/GLC_test4.csv")

# Fitting a linear regression model and storing the resulting model object into GLC_ols
GLC_ols = lm(@formula(GLC_partI ~ GLC_partII), GLC_test4) # R-style notation

# Getting the r-squared index
r2(GLC_ols)

# Creating the null model
nullmodel = lm(@formula(GLC_partI ~ 1), GLC_test4)

# Getting the fields of linear model objects.
fieldnames(StatsModels.TableRegressionModel)

# Compare the two models with the partial F-test
ft = ftest(GLC_ols.model, nullmodel.model)

