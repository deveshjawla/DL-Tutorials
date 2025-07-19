using LinearAlgebra
using Distributions

function OLSestimator(y, x)
    estimate = inv(x' * x) * (x' * y)
    return estimate
end


# Generate Data

N = 1000
K = 3
genX = MvNormal(zeros(K), ones(K))
X = rand(genX, N)
X = X'
X_noconstant = X
constant = ones(N)
X = [constant X]

genEpsilon = Normal(0, 1)
epsilon = rand(genEpsilon, N)
trueParams = [0.1, 0.5, -0.3, 0.]
Y = X * trueParams + epsilon

using Plots
plot(X, Y)
estimates = OLSestimator(Y, X)