## packages from the Julia ecosystem with Flux.
# Pkg.add("Flux")
# Pkg.add("Statistics")
# Pkg.add("Random")
# Pkg.add("Logging")
# Pkg.add("TensorBoardLogger)
# Pkg.add("ProgressMeter")
# Pkg.add("BSON")
# Pkg.add("CUDA")

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
            Dense(3, 64, relu; init=Flux.kaiming_uniform),
            Dense(64, 64, relu; init=Flux.kaiming_uniform),
            Dense(64, 64, relu; init=Flux.kaiming_uniform),
            Dense(64, 4*4*32, relu; init=Flux.kaiming_uniform),
            x->reshape(x, 4, 4, 32, :),
            ConvTranspose((3, 3), 32 => 64, relu, stride=2, pad=SamePad()),
            ConvTranspose((3, 3), 64 => 64, relu, stride=2, pad=SamePad()),
            ConvTranspose((3, 3), 64 => 32, relu, stride=2, pad=SamePad()),
            x->convert(Array{Float32}, x[2:end-1, 2:end-1, :, :]),
            ConvTranspose((1, 1), 32 => 2, identity, stride=(1, 1)),
            )
end

function get_x_data()
    train_x = CSV.read("train_x.csv", DataFrame)
    test_x = CSV.read("test_x.csv", DataFrame)
    valid_x = CSV.read("valid_x.csv", DataFrame)

    train_x = transpose(Matrix(train_x))
    test_x = transpose(Matrix(test_x))
    valid_x = transpose(Matrix(valid_x))

    train_x = convert(Array{Float32}, train_x)
    test_x = convert(Array{Float32}, test_x)
    valid_x = convert(Array{Float32}, valid_x)

    train_loader = DataLoader(train_x, batchsize=100, shuffle=false)
    test_loader = DataLoader(test_x, batchsize=20, shuffle=false)
    valid_loader = DataLoader(valid_x, batchsize=20, shuffle=false)
    return train_loader, test_loader, valid_loader
end

function reconstructionFunc(reconstruction, x)

    width = size(reconstruction)[1]
    height = size(reconstruction)[2]
    dsize = size(reconstruction)[4]

    mask1 = reshape(repeat(vcat(zeros(14, 1), ones(2, 1), zeros(14, 1)), outer=(1, dsize)), (30, 1, 1, dsize))
    mask2 = reshape(repeat(vcat(zeros(15, 1), ones(2, 1), zeros(15, 1)), outer=(1, dsize)), (32, 1, 1, dsize))

    left = cat(mask1 .* reshape(transpose(repeat(x[3, :], outer=(1, 30))), (height, 1, 1, dsize)),
           reshape(zeros(dsize*30), 30, 1, 1, dsize); dims=3)

    reconstruction = hcat(left, reconstruction)

    right = cat(reshape(reconstruction[:, end, 1, :], (30, 1, 1, dsize)),
                reshape(reconstruction[:, end, 2, :], (30, 1, 1, dsize)); dims=3)
    reconstruction = hcat(reconstruction, right)

    reconstruction = permutedims(reconstruction, [2, 1, 3, 4])

    top = cat(reshape(zeros(dsize*32), 32, 1, 1, dsize),
              mask2 .* reshape(transpose(repeat(x[1, :], outer=(1, 32))), (32, 1, 1, dsize)); dims=3)
    reconstruction = hcat(top, reconstruction)

    bottom = cat(reshape(zeros(dsize*32), 32, 1, 1, dsize),
             mask2 .* reshape(transpose(repeat(x[2, :], outer=(1, 32))), (32, 1, 1, dsize)); dims=3)

    reconstruction = hcat(reconstruction, bottom)
    reconstruction = permutedims(reconstruction, [2, 1, 3, 4])

    physics_u_advection=((reconstruction[2:end-1, 3:end, 1, :] .-
                          reconstruction[2:end-1, 1:end-2, 1, :]) .* reconstruction[2:end-1, 2:end-1, 1, :] .+
                         (reconstruction[1:end-2, 2:end-1, 1, :]  .- reconstruction[3:end, 2:end-1, 1, :])  .*
                          reconstruction[2:end-1, 2:end-1, 2, :]).* Float32(0.5)

    physics_u_diffusion = ((reconstruction[2:end-1, 1:end-2, 1, :]   .+ reconstruction[2:end-1, 3:end, 1, :] .- Float32(2.0) .*
                            reconstruction[2:end-1, 2:end-1, 1, :]) .+
                           (reconstruction[1:end-2, 2:end-1, 1, :]   .+ reconstruction[3:end, 2:end-1, 1, :] .- Float32(2.0) .*
                            reconstruction[2:end-1, 2:end-1, 1, :]))./ Float32(1.0/31.0)

    physics_v_advection = ((reconstruction[2:end-1, 3:end, 2, :]   .- reconstruction[2:end-1, 1:end-2, 2, :]) .*
                            reconstruction[2:end-1, 2:end-1, 1, :] .+
                           (reconstruction[1:end-2, 2:end-1, 2, :]  .- reconstruction[3:end, 2:end-1, 2, :])  .*
                            reconstruction[2:end-1, 2:end-1, 2, :]).* Float32(0.5)

    physics_v_diffusion = ((reconstruction[2:end-1, 1:end-2, 2, :]   .+ reconstruction[2:end-1, 3:end, 2, :] .- Float32(2.0) .*
                            reconstruction[2:end-1, 2:end-1, 2, :]) .+
                           (reconstruction[1:end-2, 2:end-1, 2, :]   .+ reconstruction[3:end, 2:end-1, 2, :] .- Float32(2.0) .*
                            reconstruction[2:end-1, 2:end-1, 2, :]))./ Float32(1.0/31.0)

    physics_continuity_u = reconstruction[2:end-1, 3:end,   1, :] .- reconstruction[2:end-1, 1:end-2, 1, :]
    physics_continuity_v = reconstruction[1:end-2, 2:end-1, 2, :] .- reconstruction[3:end, 2:end-1, 2, :]

    total_1 = cat(physics_u_advection, physics_v_advection, physics_continuity_u; dims=4)
    total_1 = permutedims(total_1, [1, 2, 4, 3])

    total_2 = cat(physics_u_diffusion, physics_v_diffusion, physics_continuity_v; dims=4)
    total_2 = permutedims(total_2, [1, 2, 4, 3])
    #physics_loss =  Flux.Losses.mse(total_1, total_2)
    #return physics_loss
    return total_1, total_2
end

function eval_loss_accuracy(loader, model, device)
    l = 0.0
    ntot = 0
    for x in loader
        x |> device
        reconstruction = model(x)
        ŷ, y = reconstructionFunc(reconstruction, x)
        l += Flux.Losses.mse(ŷ, y) * size(x)[end]
        ntot += size(x)[end]
    end
    return (loss = round(l/ntot, digits=4))
end

loss(ŷ, y) = Flux.Losses.mse(ŷ, y)

## utility functions
num_params(model) = sum(length, Flux.params(model))
round4(x) = round(x, digits=4)

# arguments for the `train` function
Base.@kwdef mutable struct Args
    η = 0.0001             # learning rate
    λ = 0                # L2 regularizer param, implemented as weight decay
    batchsize = 100      # batch size
    epochs = 5000         # number of epochs
    seed = 1             # set seed > 0 for reproducibility
    use_cuda = false      # if true use cuda (if available)
    infotime = 1 	     # report every `infotime` epochs
    checktime = 100        # Save the model every `checktime` epochs. Set to 0 for no checkpoints.
    tblogger = true      # log training with tensorboard
    dim = 3
    savepath = "runs/fdpinn_Burgers/"    # results path
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
    #train_loader, test_loader = get_data()
    train_loader, valid_loader, test_loader = get_x_data()
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
    #loss(ŷ, y) =100.0#Flux.Losses.mse(ŷ, y)

    ## TRAINING
    @info "Start Training"
    export_loss = report(0)
    history[1, 1] = 0
    history[2, 1] = export_loss[1]
    history[3, 1] = export_loss[2]
    for epoch in 1:args.epochs
        @showprogress for x in train_loader
            x = x |> device
            gs = Flux.gradient(ps) do
                reconstruction = model(x)
                ŷ, y = reconstructionFunc(reconstruction, x)
                loss(ŷ, y)
            end
            Flux.Optimise.update!(opt, ps, gs)
        end

        ## Printing and logging
        #epoch % args.infotime == 0 && report(epoch)
        # change
        if epoch % args.infotime == 0
            export_loss = report(epoch)
            history[1, epoch+1] = epoch
            history[2, epoch+1] = export_loss[1]
            history[3, epoch+1] = export_loss[2]
        end

        if args.checktime > 0 && epoch % args.checktime == 0
            !ispath(args.savepath) && mkpath(args.savepath)
            modelpath = joinpath(args.savepath, string("burger_", args.dim, "d_", epoch,".bson"))
            let model = cpu(model) #return model to cpu before serialization
                BSON.@save modelpath model epoch
            end
            @info "Model saved in \"$(modelpath)\""
        end
    end
    return history
end

history = train()


@load "runs/fdpinn_Burgers/burger_3d_3.bson" model


pred = model([0, 100, 100])

u = pred[:, :, 1, 1]
v = pred[:, :, 2, 1]
vel = (u.^2 + v.^2.0).^0.5
PlotlyJS.plot(PlotlyJS.contour(z=vel))
