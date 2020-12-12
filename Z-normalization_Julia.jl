### A Pluto.jl notebook ###
# v0.12.16

using Markdown
using InteractiveUtils

# ╔═╡ 92433940-366f-11eb-25cc-c98ee57ada7b
begin
using Statistics, StatsBase, BenchmarkTools, HypothesisTests, EffectSizes
end

# ╔═╡ 105ecc78-366f-11eb-0326-e5a81b742175
md"# Z-Score Normalization with Julia
## Benchmarking and t-testing timings
"

# ╔═╡ a06cdbe2-37fb-11eb-3978-af26ada1b99d
md""" ### Z-Score Normalization

#### What is it?

Essentially, you translate each data point in terms of the amount of standard deviations it is far from the mean of the data. The formula below, can be read as:

*Subtract each of the elements of the array `A` from the mean and divide the result by the standard deviation of the numbers in the array.*

$$\forall x_{i\ \in \ \{1,...length\{A\}\}} :\frac{x_{i} -\overline{x}}{sd}$$

#### Why use it?

* **Regression models**
  - The training of regresion models is affected by the difference in the measurement scale of the data. The value of any feature determines the update rate.
* **Distance-based methods** (e.g. KNN, K-means, PCA)
  - Each feature in the dataset is assigned equal importance and the **distance** is measured in fair terms and more efficiently and reliably."""

# ╔═╡ a0c47970-366f-11eb-14ed-f18ba6bdab8b
md" ### The `StatsBase` way"

# ╔═╡ 3c5ac99a-397a-11eb-15f5-a3e8fce6505a
begin
    dt = fit(ZScoreTransform, [0.0 -0.5 0.5; 0.0 1.0 2.0], dims=2)
    normal = StatsBase.transform(dt, [0.0 -0.5 0.5; 0.0 1.0 2.0])
	normal
end

# ╔═╡ 7193e652-3c57-11eb-1340-9f5ee7a16aa9
methods(StatsBase.transform)

# ╔═╡ a2339944-366f-11eb-0d7f-0928ee0f97fa
StatsbaseStandardize = @benchmark begin
function StatsbaseStandardize(m::Array{Float64,2})
    normal = Array{Float64,2}(undef,size(m))
    dt = fit(ZScoreTransform, m, dims=2)
    normal = StatsBase.transform(dt, m)
	return normal
end
StatsbaseStandardize([0.0 -0.5 0.5; 0.0 1.0 2.0])
end

# ╔═╡ 10f48a50-397e-11eb-051d-b980ef801ebe
md" ### Data normalization: Do It Yourself

Standardizing each data point by looping through each element of the array; first traversing through rows and then through each of the cells of the rows.

- The first step is to create an empty array of the same length with the argument of the function (i.e. `normal1`).
- Then, by using the `enumerate()` function in the two loops you can take care of the normalization of each element."

# ╔═╡ f4d00456-397d-11eb-0907-c95a54dbfd5f
begin
normal1 = Array{Float64,2}(undef, 2, 3)	
for (i,row) in enumerate(eachrow([0.0 -0.5 0.5; 0.0 1.0 2.0]))
        for (subᵢ,cell) in enumerate(row)
            normal1[i, subᵢ] = (cell - mean(row))/std(row)    
        end
    end
normal1
end

# ╔═╡ eae672ce-366f-11eb-2dac-4d4dd2b61a51
md" #### Creating and benchmarking `MyStandardize()`

The function `MyStandardize()` accepts a two-dimensional array of any length and returns its normalized version."

# ╔═╡ 0e4a9344-3670-11eb-1775-4dd6aa5afeb9
MyStandardizeArrayInit = @benchmark begin
function MyStandardize(m::Array{Float64,2})
    normal2 = Array{Float64,2}(undef, size(m))
    for (i,row) in enumerate(eachrow(m))
        for (subᵢ,cell) in enumerate(row)
            normal2[i, subᵢ] = (cell - mean(row))/std(row)    
        end
    end
    return normal2
end
MyStandardize([0.0 -0.5 0.5; 0.0 1.0 2.0])
end

# ╔═╡ a1a6d242-3670-11eb-3438-cfa59c5c95fd
md" #### Creating and benchmarking `MyStandardize2()` 

The only difference between `MyStandardize()` and `MyStandardize2()` is that the latter uses the similar() function for initializing an array `normal3`, that has the same dimensionality its elements have the same type as `m`.  
"

# ╔═╡ 40ca3572-3670-11eb-3753-c1dc93d42765
MyStandardizeSimilar = @benchmark begin
function MyStandardize2(m::Array{Float64,2})
    normal3 = similar(m)
    for (i,row) in enumerate(eachrow(m))
        for (subᵢ,cell) in enumerate(row)
           normal3[i, subᵢ] = (cell - mean(row))/std(row)    
        end
    end
    return normal3
end
MyStandardize2([0.0 -0.5 0.5; 0.0 1.0 2.0])
end

# ╔═╡ 79e68f02-3988-11eb-38a6-412eb8da3485
md" #### Creating and benchmarking the `MyStandardize!()` function 

Data standardizing by creating a mutating function; namely by transforming the data *in situ* for reducing the number of heap allocations. As expected, the mean time is better and there is one allocation less.  
"

# ╔═╡ 60b59b74-3986-11eb-3472-ff0d3c0fa6f0
MyStandardizeInSitu = @benchmark begin
function MyStandardize!(m::Array{Float64,2})
    for (i,row) in enumerate(eachrow(m))
        for (subᵢ,cell) in enumerate(row)
           m[i, subᵢ] = (cell - mean(row))/std(row)    
        end
    end
    return m
end
MyStandardize!([0.0 -0.5 0.5; 0.0 1.0 2.0])
end

# ╔═╡ 09510018-38e0-11eb-2ea2-8fc9e98ff639
md""" ### T-testing the timing differences

Conducting a series of two-sample "unpaired" or "independent" t-test or unequal variance (Welch's) t-test.

Comparing the times of the two samples: `MyStandardizeSimilar` and `MyStandardizeArrayInit`. """

# ╔═╡ 11bc618a-3966-11eb-39dd-8d3f23db7f95
md"The array with the recorded elapsed times of the experiments:"

# ╔═╡ a70c6f4a-3935-11eb-37ca-8197bdbd1243
MyStandardizeArrayInit.times

# ╔═╡ 3bd92254-3967-11eb-2591-a9f14653d8aa
md"Conducting the t-test with the `UnequalVarianceTTest()` function of the package `HypothesisTests` for comparing comparing the times of the two samples: *StatsbaseStandardize* and *MyStandardizeArrayInit*:."

# ╔═╡ a5e3edf8-368a-11eb-3767-bd2abb4a071f
UnequalVarianceTTest(StatsbaseStandardize.times, MyStandardizeArrayInit.times)

# ╔═╡ af253dea-3c4d-11eb-19f9-1bf9ae18f1bb
md" Comparing the times of the two samples: *MyStandardizeSimilar* and *MyStandardizeArrayInit*:"

# ╔═╡ a5e4b07e-368a-11eb-153e-7b13195d6f59
UnequalVarianceTTest(MyStandardizeSimilar.times, MyStandardizeArrayInit.times)

# ╔═╡ 634c2732-3c4d-11eb-380f-c1a5a87293fc
md" Comparing the times of the two samples: *MyStandardizeInSitu* and *MyStandardizeArrayInit*:"

# ╔═╡ 4779afa0-3a68-11eb-1606-89fef665e2b6
UnequalVarianceTTest(MyStandardizeInSitu.times, MyStandardizeArrayInit.times)

# ╔═╡ 38735ed6-38e0-11eb-3db0-91bdf03614dd
md" ### Measuring the effect size

Calculating the effect size with the `Cohen's D` index:

$$D = \frac{\overline{x_{1}} -\overline{x_{2}}}{sd_{pooled}},$$ where:

$$sd_{pooled} \ =\ \sqrt{\left( sd^{2}_{1} \ +sd^{2}_{2}\right) \ /\ 2}$$



Assessing the magnitude of `Cohen's D` with the following rule of thumb:
- Small effect: 0.2 - 0.5 
- Medium Effect: 0.5 - 0.8
- Large Effect: 0.8 - 1
"

# ╔═╡ a5e47298-368a-11eb-28d8-9b3ac7d0e244
CohenD(StatsbaseStandardize.times, MyStandardizeArrayInit.times)

# ╔═╡ a5ea5b84-368a-11eb-1b73-27ea432191fa
CohenD(MyStandardizeSimilar.times, MyStandardizeArrayInit.times)

# ╔═╡ 28849932-3c4d-11eb-0e84-638193c8049b
CohenD(MyStandardizeInSitu.times, MyStandardizeArrayInit.times)

# ╔═╡ Cell order:
# ╟─105ecc78-366f-11eb-0326-e5a81b742175
# ╠═92433940-366f-11eb-25cc-c98ee57ada7b
# ╟─a06cdbe2-37fb-11eb-3978-af26ada1b99d
# ╟─a0c47970-366f-11eb-14ed-f18ba6bdab8b
# ╠═3c5ac99a-397a-11eb-15f5-a3e8fce6505a
# ╠═7193e652-3c57-11eb-1340-9f5ee7a16aa9
# ╠═a2339944-366f-11eb-0d7f-0928ee0f97fa
# ╟─10f48a50-397e-11eb-051d-b980ef801ebe
# ╠═f4d00456-397d-11eb-0907-c95a54dbfd5f
# ╟─eae672ce-366f-11eb-2dac-4d4dd2b61a51
# ╠═0e4a9344-3670-11eb-1775-4dd6aa5afeb9
# ╟─a1a6d242-3670-11eb-3438-cfa59c5c95fd
# ╠═40ca3572-3670-11eb-3753-c1dc93d42765
# ╟─79e68f02-3988-11eb-38a6-412eb8da3485
# ╠═60b59b74-3986-11eb-3472-ff0d3c0fa6f0
# ╟─09510018-38e0-11eb-2ea2-8fc9e98ff639
# ╟─11bc618a-3966-11eb-39dd-8d3f23db7f95
# ╠═a70c6f4a-3935-11eb-37ca-8197bdbd1243
# ╟─3bd92254-3967-11eb-2591-a9f14653d8aa
# ╠═a5e3edf8-368a-11eb-3767-bd2abb4a071f
# ╟─af253dea-3c4d-11eb-19f9-1bf9ae18f1bb
# ╠═a5e4b07e-368a-11eb-153e-7b13195d6f59
# ╟─634c2732-3c4d-11eb-380f-c1a5a87293fc
# ╠═4779afa0-3a68-11eb-1606-89fef665e2b6
# ╟─38735ed6-38e0-11eb-3db0-91bdf03614dd
# ╠═a5e47298-368a-11eb-28d8-9b3ac7d0e244
# ╠═a5ea5b84-368a-11eb-1b73-27ea432191fa
# ╠═28849932-3c4d-11eb-0e84-638193c8049b
