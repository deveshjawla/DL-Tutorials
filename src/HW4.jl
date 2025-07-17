# Lux.jl MLP Experiments on FashionMNIST

using Lux, LuxCUDA, Random, Statistics, MLDatasets, MLUtils, Flux, Plots

# Load and preprocess FashionMNIST dataset
train_x, train_y = FashionMNIST.traindata()
test_x, test_y = FashionMNIST.testdata()

X_train = Float32.(Flux.flatten(train_x)) ./ 255
X_test = Float32.(Flux.flatten(test_x)) ./ 255
Y_train = Flux.onehotbatch(train_y .+ 1, 1:10)
Y_test = Flux.onehotbatch(test_y .+ 1, 1:10)

# 1. Vary hidden layer size and record accuracy
hidden_sizes = [10, 20, 40, 50, 100, 300]
test_accuracies = Float64[]

for hsize in hidden_sizes
    model = Chain(Dense(784 => hsize, relu), Dense(hsize => 10))
    ps, st = Lux.setup(Random.default_rng(), model)
    opt = Adam()

    for epoch in 1:10
        grads, st = Lux.gradient((ps, st), model, [(X_train, Y_train)]) do p, s, x, y
            ŷ, s = model(x, p, s)
            return logitcrossentropy(ŷ, y), s
        end
        ps = Lux.update!(opt, ps, grads[1])
    end

    ŷ, _ = model(X_test, ps, st)
    acc = mean(onecold(ŷ) .== (test_y .+ 1))
    push!(test_accuracies, acc)
end

plot(hidden_sizes, test_accuracies,
    xlabel="Hidden Layer Size", ylabel="Test Accuracy",
    title="Accuracy vs Hidden Layer Size", legend=false, marker=:circle)

# 2. Initialization impact with fixed hidden layer size = 30
init_accuracies = Float64[]

for i in 1:10
    rng = MersenneTwister(i)
    model = Chain(Dense(784 => 30, relu), Dense(30 => 10))
    ps, st = Lux.setup(rng, model)
    opt = Adam()

    for epoch in 1:10
        grads, st = Lux.gradient((ps, st), model, [(X_train, Y_train)]) do p, s, x, y
            ŷ, s = model(x, p, s)
            return logitcrossentropy(ŷ, y), s
        end
        ps = Lux.update!(opt, ps, grads[1])
    end

    ŷ, _ = model(X_test, ps, st)
    acc = mean(onecold(ŷ) .== (test_y .+ 1))
    push!(init_accuracies, acc)
end

mean_acc = mean(init_accuracies)
std_acc = std(init_accuracies)

scatter(1:10, init_accuracies, title="Accuracy Variability from Initialization",
    xlabel="Run", ylabel="Test Accuracy", marker=:circle, label="Accuracy")
hline!([mean_acc], label="Mean Accuracy", linestyle=:dash)

# 3. Train for 25 epochs, batch size 32, decaying learning rate
model = Chain(Dense(784 => 100, relu), Dense(100 => 10))
ps, st = Lux.setup(Random.default_rng(), model)
opt = Adam(0.001)

Xb, Yb = batchview((X_train, Y_train), size=32)

for epoch in 1:25
    opt.eta = 0.001 * 0.9^epoch
    for (x, y) in zip(Xb, Yb)
        grads, st = Lux.gradient((ps, st), model, [(x, y)]) do p, s, xb, yb
            ŷ, s = model(xb, p, s)
            return logitcrossentropy(ŷ, yb), s
        end
        ps = Lux.update!(opt, ps, grads[1])
    end
end

# Evaluate
ŷ, _ = model(X_test, ps, st)
acc_decay = mean(onecold(ŷ) .== (test_y .+ 1))
println("Accuracy with decay schedule: ", acc_decay)

# 4. Grid Search on Batch Size and LR Decay
batch_sizes = [16, 32, 64]
lrs = [0.001, 0.0005]
decays = [0.95, 0.9]

results = []

for bs in batch_sizes
    for lr in lrs
        for decay in decays
            model = Chain(Dense(784 => 100, relu), Dense(100 => 10))
            ps, st = Lux.setup(Random.default_rng(), model)
            opt = Adam(lr)
            Xb, Yb = batchview((X_train, Y_train), size=bs)

            for epoch in 1:20
                opt.eta = lr * decay^epoch
                for (x, y) in zip(Xb, Yb)
                    grads, st = Lux.gradient((ps, st), model, [(x, y)]) do p, s, xb, yb
                        ŷ, s = model(xb, p, s)
                        return logitcrossentropy(ŷ, yb), s
                    end
                    ps = Lux.update!(opt, ps, grads[1])
                end
            end

            ŷ, _ = model(X_test, ps, st)
            acc = mean(onecold(ŷ) .== (test_y .+ 1))
            push!(results, (bs, lr, decay, acc))
        end
    end
end

best = sort(results, by = x -> x[4], rev=true)[1]
println("Best Params => Batch Size: $(best[1]), LR: $(best[2]), Decay: $(best[3]) => Accuracy: $(best[4])")

# 5. Retrain with best params
model = Chain(Dense(784 => 100, relu), Dense(100 => 10))
ps, st = Lux.setup(Random.default_rng(), model)
opt = Adam(best[2])
Xb, Yb = batchview((X_train, Y_train), size=best[1])

for epoch in 1:25
    opt.eta = best[2] * best[3]^epoch
    for (x, y) in zip(Xb, Yb)
        grads, st = Lux.gradient((ps, st), model, [(x, y)]) do p, s, xb, yb
            ŷ, s = model(xb, p, s)
            return logitcrossentropy(ŷ, yb), s
        end
        ps = Lux.update!(opt, ps, grads[1])
    end
end

ŷ, _ = model(X_test, ps, st)
acc_final = mean(onecold(ŷ) .== (test_y .+ 1))
println("Final Accuracy after tuning: ", acc_final)