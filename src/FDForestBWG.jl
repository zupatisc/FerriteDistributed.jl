mutable struct FDForestBWG
    # Fully replicated Forest per process
    global_forest::ForestBWG
    # Communicator for sync operations
    forest_comm::MPI.Comm
    # Partitioning Algorithm for materializing
    part_algo
    cells_to_refine
end

@inline global_comm(dforest::FDForestBWG) = dforest.forest_comm

@inline global_rank(dforest::FDForestBWG) = MPI.Comm_rank(global_comm(dforest))

@inline global_nranks(dforest::FDForestBWG) = MPI.Comm_size(global_comm(dforest))

function generate_forest_grid(comm::MPI.Comm, refinement::Int64, args...; partitioning_alg = PartitioningAlgorithm.SFC())
    full_grid = generate_grid(args...)
    forest = ForestBWG(full_grid, refinement)
    refine_all!(forest, 1)
    return FDForestBWG(forest, comm, partitioning_alg)
end

function materialize_forest(dforest::FDForestBWG)
    full_grid = creategrid(dforest.global_forest)
    return NODGrid(dforest.forest_comm, full_grid, dforest.part_algo)
end

"""
        error_arr = estimate_error(forest, grid, dh, u, Cmat)
        cells_to_refine, total = dorfler_mark(error_arr, θ)
        refine!(forest, cells_to_refine)
        balanceforest!(forest)

"""
