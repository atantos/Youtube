#=
Loading the necessary packages:
GLM for fitting regression models,
DataFrames for manipulating data frames
CSV for importing/exporting csv files into/from data frames
RCall for running R code within REPL
=#
using GLM, DataFrames, CSV, RCall, StatsBase

# Reading a csv file from github into a data frame stored to the variable GLC_MLR
GLC_MLR = CSV.read(download("https://raw.githubusercontent.com/atantos/Youtube/master/GLC_test_MLR.csv"))

# Just in case you would like to see the whole content of the data frame with show().
show(GLC_MLR, allrows=true, allcols=true)

# Inspecting the data frame GLC_MLR for its dimensions.
size(GLC_MLR)

# Obtaining summary information about the entire data set.
show(describe(GLC_MLR), allrows=true, allcols=true)

# For filtering rows with "NA" strings using filter().
filter(row -> occursin("NA", row[:GR_Communication_skill]), GLC_MLR)

# Using subsetting
GLC_MLR[GLC_MLR.GR_Communication_skill .== "NA",:]

# Iterate over the symbol names and NOT the columns
function convert_NA_to_missing(dataframe_obj::DataFrame)
	for col in Symbol.(names(dataframe_obj))
	    dataframe_obj[!,col] = replace(dataframe_obj[!,col], "NA" => missing)
	end
end

convert_NA_to_missing(GLC_MLR)

show(GLC_MLR, allrows=true, allcols=true)

# Drop the rows with missing values, for now..
GLC_MLR = dropmissing(GLC_MLR)

show(GLC_MLR, allrows=true, allcols=true)

# For keeping type stability of the data frame's types.
GLC_MLR[!,:GLC_partI] = convert.(Float64, GLC_MLR[!,:GLC_partI])

show(describe(GLC_MLR), allrows=true, allcols=true)



# Fitting a linear regression model and storing the resulting model object into GLC_ols
GLC_both_numeric = lm(@formula(GLC_partI ~ GLC_partII + GLC_partIII), GLC_MLR)

fit(LinearModel,@formula(GLC_partI ~ GLC_partII + GLC_partIII), GLC_MLR)


typeof(GLC_both_numeric)
fieldnames(StatsModels.TableRegressionModel)

methodswith(StatsModels.TableRegressionModel)
methodswith(StatisticalModel)

# Predictor variables: GLC_L1, GR_Communication_skill and GLC_partI
# Response variable: GLC_partII


GLC_mixed = lm(@formula(GLC_partI ~ GLC_L1 + GLC_partIII), GLC_MLR)

GLC_mixed2 = lm(@formula(GLC_partI ~ GLC_L1 + GLC_partIII), GLC_MLR, contrasts =  Dict(:GLC_L1 => DummyCoding()))

StatsModels.ContrastsMatrix(DummyCoding(), unique(GLC_MLR.GLC_L1)).matrix


GLC_mixed_ec = lm(@formula(GLC_partI ~ GLC_L1 + GLC_partIII), GLC_MLR, contrasts = Dict(:GLC_L1 =>EffectsCoding()))


# sdiff_hypothesis should a positive definite matrix (meaning there is no collinearity)
L1_diff_hypothesis = [-1  1  0  0  0  0  0  0  0
                       0 -1  1  0  0  0  0  0  0
                       0  0 -1  1  0  0  0  0  0
                       0  0  0 -1  1  0  0  0  0
                       0  0  0  0 -1  1  0  0  0
                       0  0  0  0  0 -1  1  0  0
                       0  0  0  0  0  0 -1  1  0
                       0  0  0  0  0  0  0 -1  1];

GLC_mixed_hyp = lm(@formula(GLC_partI ~ GLC_L1 + GLC_partIII), GLC_MLR, contrasts = Dict(:GLC_L1 => HypothesisCoding(L1_diff_hypothesis)))

GLC_mixed_hyp = lm(@formula(GLC_partI ~ GLC_L1 + GLC_partIII), GLC_MLR, contrasts = Dict(:GLC_L1 => HypothesisCoding(L1_diff_hypothesis, labels=["Albanian", "Arabic", "Bulgarian", "Georgian", "Greek", "Punjabi", "Romanian", "Russian", "Turkish"])))


GLC_both_categorical = lm(@formula(GLC_partI ~ GLC_L1 + GR_Communication_skill), GLC_MLR)

L1_diff_hypothesis_GR = [-1  1  0  0
                          0 -1  1  0
                          0  0 -1  1];

GLC_both_categorical = lm(@formula(GLC_partI ~ GLC_L1 + GR_Communication_skill), GLC_MLR, contrasts = Dict(:GLC_L1 => HypothesisCoding(L1_diff_hypothesis, labels=["Albanian", "Arabic", "Bulgarian", "Georgian", "Greek", "Punjabi", "Romanian", "Russian", "Turkish"]), :GR_Communication_skill => HypothesisCoding(L1_diff_hypothesis_GR , levels = ["moderate", "none", "poor", "very good"])))

R"summary(lm(formula = wt ~ ., data = mtcars))"


# Choosing all variables and building programmatically the formula term.
# Selecting all names of the dataframe except one that is going to be our dependent variable.
predictors = filter(x -> x != :GLC_partI, names(GLC_MLR))

# Select the column with the dependent variable.
response_var = filter(x -> x == :GLC_partI, names(GLC_MLR))[1]

# Construct programmatically the formula for all predictors except the one we are interested in.
f = term(response_var) ~ sum(term.([1, (predictors...)]))

# Build the model as usual
lm(f, GLC_MLR)

# Create a macro that takes all regressors and the dependent variables and outputs a formula term.
# Mention the global scope of the dataset used.
macro dot(response_var, dataset)
	predictors = filter(x -> x != eval(response_var), names(eval(dataset)))
	response_var = filter(x -> x == eval(response_var), names(eval(dataset)))[1]
	return :(
		term(response_var) ~ sum(term.([1, ((predictors...))]))
		)
end


@dot(:GLC_partI,GLC_MLR)

lm(@dot(:GLC_partI,GLC_MLR),GLC_MLR)
