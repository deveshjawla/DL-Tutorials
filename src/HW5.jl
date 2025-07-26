# Homework 5: Convolutional Neural Networks on CIFAR-10

using Flux, MLDatasets, StatsPlots, Random, Statistics
using CSV, DataFrames
using Plots

Random.seed!(42)

#===== DATA LOADING =====#

function load_cifar10_data()
    train_x, train_y = MLDatasets.CIFAR10(split=:train)[:]
    test_x, test_y = MLDatasets.CIFAR10(split=:test)[:]
    
    train_x = Float32.(train_x) ./ 255.0f0
    test_x = Float32.(test_x) ./ 255.0f0
    
    train_y_hot = Flux.onehotbatch(train_y, 0:9)
    test_y_hot = Flux.onehotbatch(test_y, 0:9)
    
    return (train_x, train_y_hot), (test_x, test_y_hot)
end

function create_dataloader(x, y; batchsize=128, shuffle=true)
    return Flux.DataLoader((x, y); batchsize=batchsize, shuffle=shuffle)
end

function create_subset_data(train_x, train_y, n_samples)
    indices = Random.randperm(size(train_x, 4))[1:n_samples]
    return train_x[:, :, :, indices], train_y[:, indices]
end

#===== MODEL ARCHITECTURES =====#

function create_lenet(filter_size::Int=5)
    # Calculate dense layer input size
    if filter_size == 3
        dense_input = 6 * 6 * 16
    elseif filter_size == 5
        dense_input = 5 * 5 * 16
    elseif filter_size == 7
        dense_input = 3 * 3 * 16
    else
        error("Unsupported filter size: $filter_size")
    end
    
    model = Chain(
        Conv((filter_size, filter_size), 3 => 6, relu),
        MeanPool((2, 2)),
        Conv((filter_size, filter_size), 6 => 16, relu),
        MeanPool((2, 2)),
        Flux.flatten,
        Dense(dense_input => 120, relu),
        Dense(120 => 84, relu),
        Dense(84 => 10),
    )
    
    return model
end

#===== TRAINING FUNCTIONS =====#

function loss_and_accuracy(model, data_loader)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for (x, y) in data_loader
        ŷ = model(x)
        total_loss += Flux.logitcrossentropy(ŷ, y)
        total_correct += sum(Flux.onecold(ŷ, 0:9) .== Flux.onecold(y, 0:9))
        total_samples += size(y, 2)
    end
    
    avg_loss = total_loss / length(data_loader)
    accuracy = 100.0 * total_correct / total_samples
    return avg_loss, accuracy
end

function train_model(model, train_loader, test_loader, epochs; learning_rate=0.001)
    opt_rule = AdamW(learning_rate, (0.9, 0.999), 1e-4)
    opt_state = Flux.setup(opt_rule, model)
    
    history = []
    
    for epoch in 1:epochs
        for (x, y) in train_loader
            grads = Flux.gradient(m -> Flux.logitcrossentropy(m(x), y), model)
            Flux.update!(opt_state, model, grads[1])
        end
        
        if epoch % max(1, epochs ÷ 10) == 0 || epoch == epochs
            train_loss, train_acc = loss_and_accuracy(model, train_loader)
            test_loss, test_acc = loss_and_accuracy(model, test_loader)
            
            println("Epoch $epoch/$epochs - Train Acc: $(round(train_acc, digits=2))%, Test Acc: $(round(test_acc, digits=2))%")
            
            push!(history, (epoch=epoch, train_loss=train_loss, train_acc=train_acc, 
                          test_loss=test_loss, test_acc=test_acc))
        end
    end
    
    return history
end

#===== TASK 1: LENET5 ON CIFAR-10 =====#

function task1_lenet5_cifar10()
    println("Task 1: Training LeNet5 on CIFAR-10")
    
    (train_x, train_y), (test_x, test_y) = load_cifar10_data()
    train_loader = create_dataloader(train_x, train_y; batchsize=128)
    test_loader = create_dataloader(test_x, test_y; batchsize=128, shuffle=false)
    
    lenet5 = create_lenet(5)
    history = train_model(lenet5, train_loader, test_loader, 20)
    
    return lenet5, history
end

#===== TASK 2: DATASET SIZE EXPERIMENT =====#

function task2_dataset_size_experiment()
    println("Task 2: Dataset Size Experiment")
    
    (train_x, train_y), (test_x, test_y) = load_cifar10_data()
    test_loader = create_dataloader(test_x, test_y; batchsize=128, shuffle=false)
    
    experiments = [
        (n_samples=10000, epochs=6),
        (n_samples=20000, epochs=3),
        (n_samples=30000, epochs=2),
    ]
    
    results = []
    
    for (i, exp) in enumerate(experiments)
        println("Experiment $i: $(exp.n_samples) samples, $(exp.epochs) epochs")
        
        subset_x, subset_y = create_subset_data(train_x, train_y, exp.n_samples)
        train_loader = create_dataloader(subset_x, subset_y; batchsize=128)
        
        model = create_lenet(5)
        history = train_model(model, train_loader, test_loader, exp.epochs)
        
        final_test_acc = history[end].test_acc
        push!(results, (n_samples=exp.n_samples, epochs=exp.epochs, 
                       final_test_acc=final_test_acc, history=history))
        
        println("Final test accuracy: $(round(final_test_acc, digits=2))%")
    end
    
    sample_sizes = [r.n_samples for r in results]
    test_accs = [r.final_test_acc for r in results]
    
    plot_task2 = plot(sample_sizes, test_accs, 
                     marker=:circle, linewidth=2, markersize=8,
                     xlabel="Number of Training Samples", 
                     ylabel="Final Test Accuracy (%)",
                     title="Effect of Dataset Size on Performance",
                     legend=false, grid=true)
    
    folder = "results"
    isdir(folder) || mkdir(folder)
    savefig(plot_task2, joinpath(folder, "task2_dataset_size_effect.png"))
    
    return results, plot_task2
end

#===== TASK 3: FILTER SIZE COMPARISON =====#

function task3_filter_size_comparison()
    println("Task 3: Filter Size Comparison")
    
    (train_x, train_y), (test_x, test_y) = load_cifar10_data()
    train_loader = create_dataloader(train_x, train_y; batchsize=128)
    test_loader = create_dataloader(test_x, test_y; batchsize=128, shuffle=false)
    
    filter_sizes = [3, 5, 7]
    results = []
    
    for filter_size in filter_sizes
        println("Training LeNet$filter_size...")
        
        model = create_lenet(filter_size)
        history = train_model(model, train_loader, test_loader, 15)
        
        final_test_acc = history[end].test_acc
        push!(results, (filter_size=filter_size, final_test_acc=final_test_acc, 
                       history=history, model=model))
        
        println("LeNet$filter_size final test accuracy: $(round(final_test_acc, digits=2))%")
    end
    
    filter_sizes_plot = [r.filter_size for r in results]
    test_accs = [r.final_test_acc for r in results]
    
    plot_task3 = plot(filter_sizes_plot, test_accs,
                     marker=:circle, linewidth=2, markersize=8,
                     xlabel="Filter Size", 
                     ylabel="Final Test Accuracy (%)",
                     title="Effect of Filter Size on LeNet Performance",
                     legend=false, grid=true, xticks=filter_sizes_plot)
    
    folder = "results"
    isdir(folder) || mkdir(folder)
    savefig(plot_task3, joinpath(folder, "task3_filter_size_comparison.png"))
    
    return results, plot_task3
end

#===== TASK 4: FEATURE VISUALIZATION =====#

function task4_feature_investigation()
    println("Task 4: Feature Investigation")
    
    (train_x, train_y), (test_x, test_y) = load_cifar10_data()
    
    lenet3 = create_lenet(3)
    train_loader = create_dataloader(train_x, train_y; batchsize=128)
    test_loader = create_dataloader(test_x, test_y; batchsize=128, shuffle=false)
    
    history = train_model(lenet3, train_loader, test_loader, 10)
    
    sample_indices = [1, 100, 500]
    sample_images = test_x[:, :, :, sample_indices]
    
    class_names = ["airplane", "automobile", "bird", "cat", "deer", 
                   "dog", "frog", "horse", "ship", "truck"]
    
    sample_labels = [class_names[test_y[argmax(test_y[:, i]), i] + 1] for i in sample_indices]
    
    plots_list = []
    
    for (i, sample_idx) in enumerate(sample_indices)
        original = sample_images[:, :, :, i]
        
        conv1_features = lenet3[1](reshape(original, 32, 32, 3, 1))
        conv1_pool_features = lenet3[1:2](reshape(original, 32, 32, 3, 1))
        conv2_features = lenet3[1:3](reshape(original, 32, 32, 3, 1))
        
        p1 = heatmap(original[:, :, 1], title="Original ($(sample_labels[i]))", 
                     color=:gray, aspect_ratio=:equal)
        
        p2 = heatmap(conv1_features[:, :, 1, 1], title="Conv1 - Filter 1",
                     color=:viridis, aspect_ratio=:equal)
        p3 = heatmap(conv1_features[:, :, 2, 1], title="Conv1 - Filter 2", 
                     color=:viridis, aspect_ratio=:equal)
        
        p4 = heatmap(conv1_pool_features[:, :, 1, 1], title="After Pool1 - Filter 1",
                     color=:viridis, aspect_ratio=:equal)
        
        p5 = heatmap(conv2_features[:, :, 1, 1], title="Conv2 - Filter 1",
                     color=:viridis, aspect_ratio=:equal)
        p6 = heatmap(conv2_features[:, :, 5, 1], title="Conv2 - Filter 5",
                     color=:viridis, aspect_ratio=:equal)
        
        sample_plot = plot(p1, p2, p3, p4, p5, p6, layout=(2, 3), 
                          plot_title="Sample $(i): $(sample_labels[i])")
        
        push!(plots_list, sample_plot)
    end
    
    folder = "results"
    isdir(folder) || mkdir(folder)
    
    for (i, p) in enumerate(plots_list)
        savefig(p, joinpath(folder, "task4_features_sample_$(i).png"))
    end
    
    return lenet3, plots_list
end

#===== MAIN EXECUTION =====#

function run_all_tasks()
    println("Starting Homework 5: CNN Analysis on CIFAR-10")
    
    folder = "results"
    isdir(folder) || mkdir(folder)
    
    println("\nExecuting Task 1...")
    lenet5, history1 = task1_lenet5_cifar10()
    
    println("\nExecuting Task 2...")
    results2, plot2 = task2_dataset_size_experiment()
    
    println("\nExecuting Task 3...")
    results3, plot3 = task3_filter_size_comparison()
    
    println("\nExecuting Task 4...")
    lenet3_features, feature_plots = task4_feature_investigation()
    
    println("\nAll tasks completed!")
    
    return (lenet5, history1), (results2, plot2), (results3, plot3), (lenet3_features, feature_plots)
end

# Run all tasks when script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_all_tasks()
end
