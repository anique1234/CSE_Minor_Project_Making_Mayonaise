using Random
using StaticArrays
using Molly
using Molly: PairwiseInteraction, simulate!, Verlet, random_velocity
using Molly: visualize
using Unitful
using LinearAlgebra: norm
using GLMakie  # Make sure you have this loaded

function wrap_x(x::T, box_side::T) where T
    half_box = box_side / 2
    if x > half_box
        return x - box_side
    elseif x < -half_box
        return x + box_side
    else
        return x
    end
end

# --- Lennard-Jones parameters ---
ε  = 1.0          # depth of potential well
σ  = 1.0          # particle diameter (roughly)


# Custom pairwise interaction with cutoff
struct EmulsionInter{T} <: PairwiseInteraction
    cutoff::T
end

Molly.use_neighbors(::EmulsionInter) = true



# Vector force along minimum-image displacement (x periodic only)
function Molly.force(inter::EmulsionInter,
                     vec_ij,
                     atom_i::Molly.Atom,
                     atom_j::Molly.Atom,
                     force_units,
                     special,
                     coord_i,
                     coord_j,
                     boundary::Molly.RectangularBoundary,
                     velocity_i,
                     velocity_j,
                     step_n)

    # vec_ij is displacement; enforce x-periodic, y direct
    Δx = wrap_x(vec_ij[1], boundary.side_lengths[1])  # periodic in x
    Δy = wrap_x(vec_ij[2], boundary.side_lengths[2])  # periodic in y
    disp = SVector(Δx, Δy)
    r = norm(disp)
      if r == 0 || r > inter.cutoff
        return SVector(0.0, 0.0)
    end

    
    # Precompute inverse powers
    inv_r = 1/r
    sr = σ * inv_r
    sr6 = sr^6
    sr12 = sr6^2

    # Lennard-Jones force magnitude
    fmag = 24ε * inv_r * (2sr12 - sr6)

    return fmag * disp / r
end

function MyCoordinatesLogger(T, n_steps::Integer; dims::Integer=3)
    return Molly.GeneralObservableLogger(
        Molly.coordinates_wrapper,
        Array{SArray{Tuple{dims}, T, 1, dims}, 1},
        n_steps,
    )
end

MyCoordinatesLogger(n_steps::Integer; dims::Integer=3) = MyCoordinatesLogger(Float64, n_steps; dims=dims)




cutoff_tiny = 2.5*σ
temp_val = 1

L = 2.0      # box size same as boundary_tiny.side_lengths[1]
N = 4        # number of particles

function generate_atoms(n_atoms; types=[1,2], mass=1.0, boxsize=1.0, seed=nothing)
    num = n_atoms
    if seed !== nothing
        Random.seed!(seed)
    end

    # Create atoms with random types
    atoms = [Molly.Atom(mass=mass, atom_type=rand(types)) for _ in 1:n_atoms]

    # Random 2D positions
    positions = [
    SVector(rand()*2boxsize - boxsize,
            rand()*2boxsize - boxsize)
    for _ in 1:n_atoms
    ]
    return num, atoms, positions
end

num, atoms_tiny, position_tiny = generate_atoms(100; types=[1,2], mass=1.0, boxsize=0.5, seed=123)

vel_tiny = zeros(SVector{2,Float64}, num)
boundary_tiny = Molly.RectangularBoundary(L)
neighbor_finder_tiny = Molly.DistanceNeighborFinder(
    eligible=trues(num, num),
    n_steps=10,
    dist_cutoff=cutoff_tiny,
)

sys = Molly.System(
    atoms = atoms_tiny,
    coords = position_tiny,
    boundary = boundary_tiny,
    velocities = vel_tiny,
    pairwise_inters = (MyPairwiseInter=EmulsionInter(cutoff_tiny),),
    neighbor_finder = neighbor_finder_tiny,
    loggers = (coords=MyCoordinatesLogger(100,dims=2),),
    energy_units = Unitful.NoUnits,
    force_units = Unitful.NoUnits,
    k = 1.0/temp_val,
)



dt = 0.001
n_steps = 5000
verlet = Verlet(dt=dt)
simulate!(sys, verlet, n_steps);
how_many = length(atoms_tiny)
colors_tiny = fill(:blue, how_many)

visualize(
    sys.loggers.coords.history,
    sys.boundary,
    "emulsion_molly_tiny_positive_2$dt.mp4";
    color = colors_tiny,
    markersize = 0.05,
    framerate = 10,
)
