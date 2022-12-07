## packages from the Julia ecosystem with Flux.
# Pkg.add("Flux")
# Pkg.add("Statistics")
# Pkg.add("Random")
# Pkg.add("Logging")
# Pkg.add("TensorBoardLogger)
# Pkg.add("ProgressMeter")
# Pkg.add("BSON")
# Pkg.add("CUDA")

pwd()
cd("/Users/20220428_code")


using Flux
using Flux.Data: DataLoader
using Flux.Optimise: Optimiser, WeightDecay
using Flux: onehotbatch, onecold, flatten
using Flux.Losses: logitcrossentropy
using Statistics, Random
using Logging: with_logger
using TensorBoardLogger: TBLogger, tb_overwrite, set_step!, set_step_increment!
using ProgressMeter: @showprogress
import BSON
using BSON: @load
using PlotlyJS
using CUDA

using CSV
using DataFrames

function mapping()
    return Chain(
            Dense(4, 64, relu; init=Flux.kaiming_uniform),
            Dense(64, 64, relu; init=Flux.kaiming_uniform),
            Dense(64, 64, relu; init=Flux.kaiming_uniform),
            Dense(64, 4*4*32, relu; init=Flux.kaiming_uniform),
            x->reshape(x, 4, 4, 32, :),
            ConvTranspose((3, 3), 32 => 64, relu, stride=2, pad=SamePad()),
            ConvTranspose((3, 3), 64 => 64, relu, stride=2, pad=SamePad()),
            ConvTranspose((3, 3), 64 => 32, relu, stride=2, pad=SamePad()),
            x->convert(Array{Float32}, x[2:end-1, 2:end-1, :, :]),
            ConvTranspose((1, 1), 32 => 1, identity, stride=(1, 1)),
            )
end

function get_x_data(dim)
    train_num = 10000
    valid_num = 1000
    test_num = 1000

    if dim == 2
        data_x = CSV.read("data_2d.csv", DataFrame)
    elseif dim == 3
        data_x = CSV.read("data_3d.csv", DataFrame)
    elseif dim == 4
        data_x = CSV.read("data_4d.csv", DataFrame)
    end

    data_x = transpose(Matrix(data_x))
    data_x = vcat(data_x, zeros(4 - dim, size(data_x)[2]))

    ### CPU
    data_x = convert(Array{Float32}, data_x)

    train_x = data_x[:, 1:10000]
    valid_x = data_x[:, 10001:11000]
    test_x = data_x[:, 11001:12000]

    # Create two DataLoader objects (mini-batch iterators)
    train_loader = DataLoader(train_x, batchsize=100, shuffle=false)
    valid_loader = DataLoader(valid_x, batchsize=20, shuffle=false)
    test_loader = DataLoader(test_x, batchsize=20, shuffle=false)
    return train_loader, valid_loader, test_loader
end

function get_xy_data(dim)
    train_num = 10000
    valid_num = 1000
    test_num = 1000

    if dim == 2
        data_x = CSV.read("data_2d.csv", DataFrame)
        data_y = CSV.read("train_data_y_2d.csv", header=0, DataFrame)
    elseif dim == 3
        data_x = CSV.read("data_3d.csv", DataFrame)
        data_y = CSV.read("train_data_y_3d.csv", header=0, DataFrame)
    elseif dim == 4
        data_x = CSV.read("data_4d.csv", DataFrame)
        data_y = CSV.read("train_data_y_4d.csv", header=0, DataFrame)
    end

    data_x = transpose(Matrix(data_x))
    data_x = vcat(data_x, zeros(4 - dim, size(data_x)[2]))
    data_y = reshape(Matrix(data_y), 30, 30, 1, :)

    ### CPU
    data_x = convert(Array{Float32}, data_x)
    data_y = convert(Array{Float32}, data_y)

    train_x = data_x[:, 1:10000]
    train_y = data_y[:, :, :, 1:10000]
    valid_x = data_x[:, 10001:11000]
    valid_y = data_y[:, :, :, 10001:11000]
    test_x = data_x[:, 12001:13000]
    test_y = data_y[:, :, :, 12001:13000]

    # Create two DataLoader objects (mini-batch iterators)
    train_loader = DataLoader((train_x, train_y), batchsize=100, shuffle=false)
    valid_loader = DataLoader((valid_x, valid_y), batchsize=20, shuffle=false)
    test_loader = DataLoader((test_x, test_y), batchsize=20, shuffle=false)
    return train_loader, valid_loader, test_loader
end


loss(ŷ, y) = Flux.Losses.mse(ŷ, y)

function reconstructionFunc(reconstruction, x)
    width = size(reconstruction)[1]
    height = size(reconstruction)[2]
    dsize = size(reconstruction)[4]

    reconstruction = reshape(reconstruction, (width, height, dsize))

    left = repeat(x[3, :], outer=(1, 30))
    left = transpose(left)
    left = reshape(left, (height, 1, dsize))
    reconstruction = hcat(left, reconstruction)

    right = repeat(x[4, :], outer=(1, 30))
    right = transpose(right)
    right = reshape(right, (height, 1, dsize))
    reconstruction = hcat(reconstruction, right)

    top = repeat(x[1, :], outer=(1, 32))
    top = transpose(top)
    top = reshape(top, (1, width+2, dsize))
    reconstruction = vcat(top, reconstruction)

    bottom = repeat(x[2, :], outer=(1, 32))
    bottom = transpose(bottom)
    bottom = reshape(bottom, (1, width+2, dsize))
    reconstruction = vcat(reconstruction, bottom)

    reconstruction = permutedims(reconstruction, [2, 1, 3])

    mat_top = reconstruction[2:end-1, 3:end, :]
    mat_bottom = reconstruction[2:end-1, 1:end-2, :]
    mat_left = reconstruction[1:end-2, 2:end-1, :]
    mat_right = reconstruction[3:end, 2:end-1, :]
    mat_mid = reconstruction[2:end-1, 2:end-1, :]

    physics = mat_top .+ mat_bottom .+ mat_left .+ mat_right .- Float32(4.0) .* mat_mid
    return physics
end

function eval_loss_accuracy(loader, model, device)
    l = 0.0
    ntot = 0
    for x in loader
        x |> device
        reconstruction = model(x)
        ŷ = reconstructionFunc(reconstruction, x)
        l += sum((ŷ).^2)/length(ŷ) * size(x)[end]
        ntot += size(x)[end]
    end
    return (loss = round(l/ntot, digits=4))
end

## utility functions
num_params(model) = sum(length, Flux.params(model))
round4(x) = round(x, digits=4)

# arguments for the `train` function
Base.@kwdef mutable struct Args
    η = 0.001             # learning rate
    λ = 0                # L2 regularizer param, implemented as weight decay
    batchsize = 100      # batch size
    epochs = 100         # number of epochs
    seed = 1             # set seed > 0 for reproducibility
    use_cuda = false      # if true use cuda (if available)
    infotime = 1 	     # report every `infotime` epochs
    checktime = 10        # Save the model every `checktime` epochs. Set to 0 for no checkpoints.
    tblogger = true      # log training with tensorboard
    dim = 2
    savepath = "runs/fdpinn2d_epoch100/"    # results path
end

function train(; kws...)
    args = Args(; kws...)
    args.seed > 0 && Random.seed!(args.seed)

    use_cuda = args.use_cuda && CUDA.functional()

    if use_cuda
        device = gpu
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end


    ## DATA
    train_loader, valid_loader, test_loader = get_x_data(args.dim)
    ## MODEL AND OPTIMIZER
    model = mapping() |> device

    ps = Flux.params(model)

    opt = ADAM(args.η, (0.9, 0.999))

    ## LOGGING UTILITIES
    if args.tblogger
        tblogger = TBLogger(args.savepath, tb_overwrite)
        set_step_increment!(tblogger, 0) # 0 auto increment since we manually set_step!
        @info "TensorBoard logging at \"$(args.savepath)\""
    end

    function report(epoch)
        train = eval_loss_accuracy(train_loader, model, device)
        test = eval_loss_accuracy(test_loader, model, device)

        println("Epoch: $epoch   Train: $(train)   Test: $(test)")
        return (train, test)
    end

    ## add
    history = Array{Float32}(undef, 3, args.epochs+1)

    ## TRAINING
    @info "Start Training"
    export_loss = report(0)
    history[1, 1] = 0
    history[2, 1] = export_loss[1]
    history[3, 1] = export_loss[2]
    for epoch in 1:args.epochs
        @showprogress for x in train_loader
            ###
            x = x |> device
            gs = Flux.gradient(ps) do
                    reconstruction = model(x)
                    ŷ = reconstructionFunc(reconstruction, x)
                    y = zeros(Float32, 30, 30, 100)
                    ### y = CUDA.zeros(Float32, 30, 30)
                    loss(ŷ, y)
                end

            Flux.Optimise.update!(opt, ps, gs)
        end

        ## Printing and logging
        if epoch % args.infotime == 0
            export_loss = report(epoch)
            history[1, epoch+1] = epoch
            history[2, epoch+1] = export_loss[1]
            history[3, epoch+1] = export_loss[2]
        end

        if args.checktime > 0 && epoch % args.checktime == 0
            !ispath(args.savepath) && mkpath(args.savepath)
            modelpath = joinpath(args.savepath, string("pinn_model_", args.dim, "d_", epoch,".bson"))
            let model = cpu(model) #return model to cpu before serialization
                BSON.@save modelpath model epoch
            end
            @info "Model saved in \"$(modelpath)\""
        end
    end
    return history
end




history = train()


@load "runs/fdpinn2d_epoch100/pinn_model_2d_100.bson" model
dimension = 2
#train_loader, valid_loader, test_loader = get_x_data(dimension)
train_loader, valid_loader, test_loader = get_xy_data(dimension)
data = first(train_loader)

id = 1
#x = data[:, id]
x = data[1][:, id]
y = data[2][:, :, 1, id]
#y = zeros(30, 30)
pred = model(x)[:, :, 1, 1]

diff = abs.(pred .- y)
PlotlyJS.plot(PlotlyJS.contour(z=pred))
PlotlyJS.plot(PlotlyJS.contour(z=y))

p1 = PlotlyJS.plot(PlotlyJS.contour(z=pred), Layout(title="Precition"))
p2 = PlotlyJS.plot(PlotlyJS.contour(z=y), Layout(title="True value"))
p3 = PlotlyJS.plot(PlotlyJS.contour(z=diff), Layout(title="Difference"))

p = [p1 p2 p3]

relayout!(p, height=250, width=500, title_text="Finite Difference Physics Inform Neural Network")

# write data
@load "runs/fdpinn2d_epoch100/pinn_model_2d_100.bson" model
train_loader, valid_loader, test_loader = get_xy_data(dimension)
#train_loader, valid_loader, test_loader = get_x_data(dimension)
data = first(test_loader)
data[1]
data[2]

df_x = DataFrame()
df_y = DataFrame()
df_pred = DataFrame()
#id = 1
for id in 1:20
    x = data[1][:, id]
    y = data[2][:, :, 1, id]
    pred = model(x)[:, :, 1, 1]

    x = DataFrame(reshape(x, 1, 4), :auto)
    y = DataFrame(reshape(y, 1, 900), :auto)
    pred = DataFrame(reshape(pred, 1, 900), :auto)
    df_x = vcat(df_x, x)
    df_y = vcat(df_y, y)
    df_pred = vcat(df_pred, pred)
end

# output
# history
filename = "fdpinn2d_history.csv"
CSV.write(filename, DataFrame(transpose(history), :auto), writeheader=true)
# x
filename = "fdpinn2d_x.csv"
CSV.write(filename, df_x, writeheader=true)
# y
filename = "fdpinn2d_y.csv"
CSV.write(filename, df_y, writeheader=true)
# pred
filename = "fdpinn2d_pred.csv"
CSV.write(filename, df_pred, writeheader=true)
