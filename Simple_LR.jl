# Loading the necessary packages
# Package for fitting regression models
using GLM

# Package for manipulating data frames
using DataFrames

# Package for importing/exporting csv files into/from data frames
using CSV

# Reading a csv file from github into a data frame stored to the variable GLC_test4
GLC_test4 = CSV.read(download("https://raw.githubusercontent.com/atantos/Youtube/master/GLC_test4.csv"))

# Reading a local csv file
GLC_test4 = CSV.read("<path_to_file>/GLC_test4.csv")

# Creating a function for converting commas into dots
function comma_rep(x)
    replace(x, r"," => ".")
end

# Creating a function for converting strings into floats
function convert_str_float64(x)
    parse(Float64, x)
end

# Applying the previously stored and loaded functions on the cells of GLC_test4
for x in Symbol.(names(GLC_test4))
    GLC_test4[!,x] = map(comma_rep, GLC_test4[!,x])
    GLC_test4[!,x] = map(convert_str_float64, GLC_test4[!,x])
end

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

