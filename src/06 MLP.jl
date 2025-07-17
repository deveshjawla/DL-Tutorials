# Classification of MNIST dataset using an MLP and Lux.jl

#Stochastic Gradient Descent

using Lux, MLUtils, Optimisers, OneHotArrays, Random, Statistics, Printf, Zygote, JLD2, Plots
using CSV, DataFrames
rng = Xoshiro(1)
function flatten(x::AbstractArray)
    return reshape(x, :, size(x)[end])
end

train = CSV.read("src/mnist/mnist_train.csv", DataFrame, header=1)
test = CSV.read("src/mnist/mnist_test.csv", DataFrame, header=1)

function mnistloader(data::DataFrame, batch_size_)
    x4dim = reshape(permutedims(Matrix{Float32}(select(data, Not(:label)))), 28, 28, 1, :)   # insert trivial channel dim
    x4dim = mapslices(x -> reverse(permutedims(x ./ 255), dims=1), x4dim, dims=(1, 2)) #standardize the data, we divide by the largest value. you can check that it correctly odes the job.
    x4dim = meanpool((x4dim), (2, 2)) #this is being done from experience to reduce dimensionality, you can do it by trial and error for other problems also. but we make 75% efficiency just by doing this in this problem.
    x4dim = flatten(x4dim)
    # ys = permutedims(data.label) .+ 1

    yhot = onehotbatch(Vector(data.label), 0:9)  # make a 10×60000 OneHotMatrix, you can try without doing this, and see if the model learns anything.
    return DataLoader((x4dim, yhot); batchsize=batch_size_, shuffle=true)
    # return x4dim, ys
end

x1, y1 = first(mnistloader(train, 128)) #batch size is a hyper-parameter

#===== MODEL =====#

model = Chain(
    Dense(196 => 14, relu), # the number of layers, activation functions, and the number of neurons per layer are all hyper-parameters.
    Dense(14 => 14, relu),
    Dense(14 => 10),
) #MLP, FFNN, DNN ∈ ANN, Vanilla Neural Network

#===== METRICS =====#

const lossfn = CrossEntropyLoss(; logits=Val(true))

import Term: tprintln
using Term.TermMarkdown
using Markdown

tprintln(md"""
$mean(-\sum y log(ŷ) + \epsilon))$
""")

function accuracy_score(model, ps, st, loader)
    st = Lux.testmode(st)                         # disable dropout, etc.
    correct = 0; total = 0
    for (x, y) in loader
        ŷ = onecold(Array(first(model(x, ps, st))), 0:9)
        yᵗ = onecold(y, 0:9)
        correct += sum(ŷ .== yᵗ)
        total   += length(yᵗ)
    end
    return correct / total
end

#===== TRAINING =====#

train_dataloader, test_dataloader = mnistloader(train, 512), mnistloader(test, 10000) #batch size also is a hyper-parameter
ps, st = Lux.setup(rng, model) # model = arch + parameters + (optionally) state

vjp = AutoZygote() # AD backend

train_state = Training.TrainState(model, ps, st, AdamW(lambda=3e-4)) # optimizer

mkpath("./mnist/Lux MLP trained models")
### Lets train the model
nepochs = 10
train_accuracy, test_accraucy = 0.1, 0.1

save_dir = joinpath("mnist", "trained_models")
mkpath(save_dir)

for epoch in 1:nepochs
    t_start = time()
    for (x, y) in train_dataloader
        # train_state = single_train_step!(vjp, lossfn, (x, y), train_state)
        _, _, _, train_state = Training.single_train_step!(
            vjp, lossfn, (x, y), train_state,
        )
    end
    epoch_time = time() - t_start

    train_acc = accuracy_score(model, train_state.parameters, train_state.states, train_dataloader) * 100
    test_acc  = accuracy_score(model, train_state.parameters, train_state.states, test_dataloader ) * 100

    @printf "[%2d/%2d]  %.2fs  train: %.2f%%  test: %.2f%%\n" epoch nepochs epoch_time train_acc test_acc

    if epoch % 5 == 0
        checkpoint = joinpath(save_dir, "MLP_epoch_$(lpad(epoch,2,'0')).jld2")
        JLD2.save(
            checkpoint,                # ← function API avoids macro issues
            "trained_parameters", train_state.parameters,
            "trained_states",     train_state.states
        )
        println("  ↳ checkpoint saved → " * checkpoint)
    end
end
# HW TODO is to load trained models and use them for testing on the test dataset again and calcualte the balanced_accuracy instead of Accuracy.

function balanced_accuracy(model, ps, st, loader)
    st = Lux.testmode(st)
    preds = Int[]; targets = Int[]
    for (x, y) in loader
        append!(preds  , onecold(Array(first(model(x, ps, st))), 0:9))
        append!(targets, onecold(y, 0:9))
    end
    recalls = [sum((preds .== c) .& (targets .== c)) / max(sum(targets .== c), 1)
        for c in 0:9]
    return mean(recalls)
end

for epoch in 5:5:nepochs
    checkpoint = joinpath(save_dir, "MLP_epoch_$(lpad(epoch,2,'0')).jld2")
    if isfile(checkpoint)
        data = JLD2.load(checkpoint)                    # returns Dict{String,Any}
        bal_acc = balanced_accuracy(model, data["trained_parameters"], data["trained_states"], test_dataloader) * 100
        @printf "Balanced accuracy (epoch %2d): %.2f%%
" epoch bal_acc
    end
end