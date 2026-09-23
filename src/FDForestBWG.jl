mutable struct FDForestBWG
    # Fully replicated Forest per process
    replicated_forest::ForestBWG
    # Communicator for sync operations
    forest_comm::MPI.Comm
    # Partitioning Algorithm for materializing
    part_algo::Union{PartitioningAlgorithm.SFC, PartitioningAlgorithm.Metis}
end

@inline global_comm(dforest::FDForestBWG) = dforest.forest_comm

@inline global_rank(dforest::FDForestBWG) = MPI.Comm_rank(global_comm(dforest)) + 1

@inline global_nranks(dforest::FDForestBWG) = MPI.Comm_size(global_comm(dforest))

function generate_forest_grid(comm::MPI.Comm, refinement::Int64, args...; partitioning_alg = PartitioningAlgorithm.SFC())
    full_grid = generate_grid(args...)
    forest = ForestBWG(full_grid, refinement)
    #refine_all!(forest, 1)
    return FDForestBWG(forest, comm, partitioning_alg)
end

function generate_forest_grid(comm::MPI.Comm, refinement::Int64, base_grid::Grid; partitioning_alg = PartitioningAlgorithm.SFC())
    forest = ForestBWG(base_grid, refinement)
    #refine_all!(forest, 1)
    return FDForestBWG(forest, comm, partitioning_alg)
end

"""
Return a NODGrid with the local grid
"""
function creategrid(dforest::FDForestBWG)
    return NODGrid(dforest)
end

function refine!(dforest::FDForestBWG, cells_to_refine::Vector{Int})
    # sync cells_to_refine here and refine
    local_cells_len = length(cells_to_refine)

    global_cells_len = MPI.Allgather(local_cells_len, dforest.forest_comm)
    global_cells_to_refine = similar(cells_to_refine, sum(global_cells_len))

    vbuf = MPI.VBuffer(global_cells_to_refine, global_cells_len)
    MPI.Allgatherv!(cells_to_refine, vbuf, dforest.forest_comm)

    refine!(dforest.replicated_forest, global_cells_to_refine)

    return 0
end

function balanceforest!(dforest::FDForestBWG)
    return Ferrite.balanceforest!(dforest.replicated_forest)
end
# """
#         error_arr = estimate_error(forest, grid, dh, u, Cmat)
#         cells_to_refine, total = dorfler_mark(error_arr, θ)
#         refine!(forest, cells_to_refine)
#         balanceforest!(forest)
#
# """
