"""
A Dedalus v3 script for running 3D numerical simulations of in a Cartesian box. This script
allows for an internal heating function, such as in Currie et al. 2020.
To Do:

Usage:
    d3_rb_convect.py [options]
    d3_rb_convect.py [--currie | --kazemi] [options]

Options:
    --Ra=<Ra>                       # Rayleigh number
    --Pr=<Pr>                       # Prandtl number [default: 1]
    --Ta=<Ta>                       # Taylor number
    --theta=<theta>                 # co-latitude of box to rotation vector [default: 45]
    --Ly=<Ly>                       # Aspect Ratio of the box [default: 1]
    --Lz=<Lz>                       # Depth of the box [default: 1]
    --Ny=<Ny>                       # Horizontal resolution [default: 128]
    --Nz=<Nz>                       # Vertical resolution [default: 256]
    --tau=<tau>                     # timescale [default: viscous]
    --maxdt=<maxdt>                 # Maximum timestep [default: 1e-5]
    --stop=<stop>                   # Simulation stop time [default: 1.0]
    --currie                        # Run with Currie 2020 heating function
    --kazemi                        # Run with Kazemi 2022 heating function
    --Hwidth=<Hwidth>               # Width of the heating zone [default: 0.2]
    --slip=SLIP                     # Boundary conditions No/Free [default: free]
    --top=TOP                       # Top boundary condition [default: insulating]
    --bottom=BOTTOM                 # Bottom boundary condition [default: insulating]
    --snaps=<snaps>                 # Snapshot interval [default: 500]
    --slices=<slices>               # Slice interval [default: 250]
    --horiz=<horiz>                 # Horizontal analysis interval [default: 100]
    --scalar=<scalar>               # Scalar analysis interval [default: 1]
    -o OUT_PATH, --output OUT_PATH  # output file [default: DATA/output/]
    -i IN_PATH, --input IN_PATH     # path to read in initial conditions from
    -m=<mesh>, --mesh=<mesh>        # Processor Mesh
    -t --test                       # Do not save any output
    -k, --kill                      # Kills the program after building the solver.
    -f, --function                  # Plots the heating function
"""

import numpy as np
import dedalus.public as d3
import logging
import os
import pathlib
from glob import glob
from docopt import docopt
import json
from mpi4py import MPI

ncpu = MPI.COMM_WORLD.size


logger = logging.getLogger(__name__)


class NaNFlowError(Exception):
    exit_code = -50
    pass


def argcheck(argument, params, type=float):
    if argument:
        return type(argument)
    else:
        return params


exit_code = 0
args = docopt(__doc__, version="0.1")

mesh = args["--mesh"]
if mesh is not None:
    mesh = mesh.split(",")
    mesh = [int(mesh[0]), int(mesh[1])]
logger.info("ncpu = {}".format(ncpu))
log2 = np.log2(ncpu)
if log2 == int(log2):
    mesh = [int(2 ** np.ceil(log2 / 2)), int(2 ** np.floor(log2 / 2))]
logger.info("running on processor mesh={}".format(mesh))

if not (args["--test"]):
    outpath = os.path.normpath(args["--output"]) + "/"
    os.makedirs(outpath, exist_ok=True)
    logger.info("Writing to {}".format(outpath))

if args["--input"]:
    restart_path = os.path.normpath(args["--input"]) + "/"
    logger.info("Reading from {}".format(restart_path))

Ly = int(args["--Ly"])
Lz = int(args["--Lz"])
if args["--input"]:
    with open(restart_path + "run_params/runparams.json", "r") as f:
        inparams = json.load(f)
    Ny = inparams["Ny"]
    Nz = inparams["Nz"]
else:
    Nz = int(args["--Nz"])
    if args["--Ny"]:
        Ny = int(args["--Ny"])
    else:
        Ny = 2 * Nz

# Ra = argcheck(args["--Ra"], rp.Ra)
Ra = float(args["--Ra"])
# Pr = argcheck(args["--Pr"], rp.Pr)
Pr = float(args["--Pr"])
# Ta = argcheck(args["--Ta"], rp.Ta)
Ta = float(args["--Ta"])

logger.info(f"Ro_c = {np.sqrt(Ra / (Pr * Ta)):1.2e}")

# snapshot_iter = argcheck(args["--snaps"], rp.snapshot_iter, type=int)
snapshot_iter = int(args["--snaps"])
# slices_iter = argcheck(args["--slices"], rp.slices_iter, type=int)
slices_iter = int(args["--slices"])
# horiz_iter = argcheck(args["--horiz"], rp.horiz_iter, type=int)
horiz_iter = int(args["--horiz"])
# scalar_iter = argcheck(args["--scalar"], rp.scalar_iter, type=int)
scalar_iter = int(args["--scalar"])

if args["--kazemi"]:
    heat_type = "Kazemi"
elif args["--currie"]:
    heat_type = "Currie"
else:
    heat_type = None
if args["--slip"] == "no":
    slip_type = "No Slip"
else:
    slip_type = "Free Slip"

logger.info(
    f"Ra={Ra:1.1e}, Pr={Pr:1.1e}, Ta={Ta:1.1e}\nLy={Ly}, Lz={Lz}, Ny={Ny}, Nz={Nz}, Heated={heat_type}, {slip_type}"
)

# parallel = "gather"
parallel = None

# ====================
# SET UP PROBLEM
# ====================
dealias = 3 / 2
dtype = np.float64
timestepper = d3.SBDF2  # Change timestepper from RK443 to lower memory usage

# stop_sim_time = argcheck(args["--stop"], rp.stop_sim_time, type=float)
stop_sim_time = float(args["--stop"])
# stop_wall_time = rp.stop_wall_time
# stop_iteration = rp.end_iteration

# max_timestep = argcheck(args["--maxdt"], rp.max_timestep, type=float)
max_timestep = float(args["--maxdt"])
logger.info(f"max_timestep = {max_timestep}")

# ===Initialise basis===
coords = d3.CartesianCoordinates("x", "y", "z")
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords["x"], size=Ny, bounds=(0, Ly), dealias=dealias)
ybasis = d3.RealFourier(coords["y"], size=Ny, bounds=(0, Ly), dealias=dealias)
zbasis = d3.ChebyshevT(coords["z"], size=Nz, bounds=(0, Lz), dealias=dealias)
x, y, z = dist.local_grids(xbasis, ybasis, zbasis)
all_bases = (xbasis, ybasis, zbasis)
hor_bases = (xbasis, ybasis)

# Add fields (e.g. variables of the equations)
# Velocity
u = dist.VectorField(coords, name="u", bases=all_bases)
# Pressure
p = dist.Field(name="p", bases=all_bases)
# Temperature
Temp = dist.Field(name="Temp", bases=all_bases)

# Add Tau Terms
# Velocity tau terms, tau_u1 = (tau_1, tau_2)
tau_u1 = dist.VectorField(coords, name="tau_u1", bases=hor_bases)
tau_u2 = dist.VectorField(coords, name="tau_u2", bases=hor_bases)
# Temperature Tau Terms
tau_T3 = dist.Field(name="tau_T3", bases=hor_bases)
tau_T4 = dist.Field(name="tau_T4", bases=hor_bases)
# Scalar tau term for pressure gauge fixing
tau_p = dist.Field(name="tau_p")

# Substitutions
x_hat, y_hat, z_hat = coords.unit_vector_fields(dist)
lift_basis = zbasis.derivative_basis(1)  # Chebyshev U Basis
lift = lambda A: d3.Lift(A, lift_basis, -1)  # Shortcut for multiplying by U_{N-1}(y)
uz = d3.Differentiate(u, coords["z"])
Tz = d3.Differentiate(Temp, coords["z"])

u_x = u @ x_hat
u_y = u @ y_hat
u_z = u @ z_hat
dzu_y = d3.Differentiate(u_y, coords["z"])
dzu_x = d3.Differentiate(u_x, coords["z"])


f_cond = -d3.Differentiate(Temp, coords["z"])
f_conv = u_z * Temp
g_operator = d3.grad(u) - z_hat * lift(tau_u1)
h_operator = d3.grad(Temp) - z_hat * lift(tau_T3)
F = 1

# Add coriolis term
Tah = np.sqrt(Ta)
theta_deg = float(args["--theta"])
theta = theta_deg * np.pi / 180
# rotation vector
omega = dist.VectorField(coords, name="omega", bases=all_bases)
# omega vector, for cartesian is (0, sin(theta), cos(theta)),
# for spherical is (cos(theta), sin(theta), 0)
# where theta is the co-latitude of the box
omega["g"][0] = 0
omega["g"][1] = np.sin(theta)
omega["g"][2] = np.cos(theta)

# #? =================
# #! HEATING FUNCTION
# #? =================
# Following Currie et al. 2020 Set-up B
# Width of middle 'convection zone' with no heating/cooling
heating_width = float(args["--Hwidth"])

H = convection_height = Lz / (1 + 2 * heating_width)
# Width of heating and cooling layers
Delta = heating_width * H

heat = dist.Field(bases=zbasis)
if args["--currie"]:
    heat_func = lambda z: (F / Delta) * (
        1 + np.cos((2 * np.pi * (z - (Delta / 2))) / Delta)
    )
    cool_func = lambda z: (F / Delta) * (
        -1 - np.cos((2 * np.pi * (z - Lz + (Delta / 2))) / Delta)
    )

    heat["g"] = np.piecewise(
        z, [z <= Delta, z >= Lz - Delta], [heat_func, cool_func, 0]
    )
elif args["--kazemi"]:
    l = 0.1
    beta = 1
    a = 1 / (0.1 * (1 - np.exp(-1 / l)))
    heat_func = lambda z: a * np.exp(-z / l) - beta
    heat["g"] = heat_func(z)
else:
    #! === No Heating ===
    heat["g"] = np.zeros(heat["g"].shape)

if args["--function"]:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(heat["g"], z, c="k", s=5)
    if args["--currie"]:
        ax.axhspan(0, Delta, color="r", alpha=0.2)
        ax.text(np.min(heat["g"]), 0.05, "Heating Zone", color="r")
        ax.axhspan(0.5 - (H / 2), 0.5 + (H / 2), color="k", alpha=0.2)
        ax.text(np.min(heat["g"]), 0.5, "Convection Zone", color="k")
        ax.axhspan(Lz - Delta, 1, color="blue", alpha=0.2)
        ax.text(0.4 * np.max(heat["g"]), 0.95, "Cooling Zone", color="blue")
        ax.set_xlabel("Heat")
        ax.set_ylabel("z")
        ax.set_title("Currie Heat Function")
    if args["--kazemi"]:
        line = -l * np.log(beta / a)
        ax.axhspan(0, line, color="r", alpha=0.2)
        ax.axhspan(line, 1, color="blue", alpha=0.2)
        ax.text(8, 0.1, "Heating Zone", ha="center", color="r")
        ax.text(8, 0.6, "Cooling Zone", ha="center", color="blue")
        ax.set_xlabel("Heat")
        ax.set_ylabel("z")
        ax.set_title("Kazemi Heat Function")
    if not args["--test"]:
        fig.savefig(outpath + "heat_func.pdf")
    else:
        fig.savefig("heat_func.pdf")
        exit(0)

# === Initialise Problem ===
problem = d3.IVP(
    [u, p, Temp, tau_u1, tau_u2, tau_T3, tau_T4, tau_p], time="t", namespace=locals()
)
# Thermal diffusion time
if args["--tau"] == "thermal":
    # Mass Conservation
    problem.add_equation("trace(g_operator) + tau_p= 0")  # needs a gauge fixing term
    # Momentum Equation
    problem.add_equation(
        "dt(u) - Pr * (div(g_operator)) + grad(p) - (Ra * Pr)*Temp*z_hat + lift(tau_u2) = - u@g_operator - (Tah / Pr) * cross(omega, u)"
    )
    # Temp Evolution
    problem.add_equation(
        "dt(Temp) + lift(tau_T4) -  (div(h_operator)) = -(u@h_operator) + heat"
    )
# Viscous diffusion time
elif args["--tau"] == "viscous":
    # Mass Conservation
    problem.add_equation("trace(g_operator) + tau_p= 0")  # needs a gauge fixing term
    # Momentum Equation
    problem.add_equation(
        "dt(u) - (div(g_operator)) + grad(p) - (Ra / Pr)*Temp*z_hat + lift(tau_u2) = - u@g_operator - Tah*cross(omega, u)"
    )
    # Temp Evolution
    problem.add_equation(
        "dt(Temp) + lift(tau_T4) - (1 / Pr) * (div(h_operator)) = -(u@h_operator) + (1 / Pr) * heat"
    )
else:
    raise ValueError(
        f'Invalid tau value {args["--tau"]}. Must be "viscous" or "thermal".'
    )

# ? === Driving Boundary Conditions ===
#! === Boundary Driven ===
# * === RB1 (Temp gradient)===
# # T=0 at top, T=1 at bottom
# problem.add_equation("Temp(z=0) = 1")
# problem.add_equation("Temp(z=Lz) = 0")

# * === RB2 (fixed flux) ===
# # Goluskin 2015
# problem.add_equation('Tz(z=0) = -F')
# problem.add_equation('Tz(z=Lz) = -F')

# *=== RB3 ===*
# # Fixed F at bottom, T=0 at top
# problem.add_equation('Tz(z=0) = -F')
# problem.add_equation('Temp(z=Lz) = 0')

#! === Internally Heated ===
# * === IH1 (T=0) ===
# # T=0 at top and bottom (Goluskin & van der Poel 2016)
# problem.add_equation('Temp(z=0) = 0')
# problem.add_equation('Temp(z=Lz) = 0')

# * === IH2 ===
# # Insulating bottom, fixed flux top
# problem.add_equation('Tz(z=0) = 0')
# problem.add_equation('Tz(z=Lz) = -F')
# if args["--currie"] or args["--kazemi"]:
#     if args["--ff"]:
#         # Insulating Top and Bottom
#         problem.add_equation("Tz(z=0) = 0")
#         problem.add_equation("Tz(z=Lz) = 0")
#     else:
#         # * === IH3 ===
#         # # Kazemi et al. 2022
#         # # Insulating bottom, T=0 top
#         problem.add_equation("Tz(z=0) = 0")
#         problem.add_equation("Temp(z=Lz) = 0")
# else:
#     if args["--ff"]:
#         problem.add_equation("Tz(z=0) = -F")
#         problem.add_equation("Tz(z=Lz) = 0")
#     else:
#         problem.add_equation("Tz(z=0) = -F")
#         problem.add_equation("Temp(z=Lz) = 0")

if args["--top"] == "insulating":
    problem.add_equation("Tz(z=Lz) = 0")
    boundary_conditions = "Insulating top"
elif args["--top"] == "vanishing":
    problem.add_equation("Temp(z=Lz) = 0")
    boundary_conditions = "Vanishing top"
elif args["--top"] == "fixed_flux":
    problem.add_equation("Tz(z=Lz) = -F")
    boundary_conditions = "Fixed Flux top"
else:
    raise ValueError(
        f"Invalid top boundary condition {args['--top']}, must be 'insulating', 'vanishing' or 'fixed_flux'"
    )

if args["--bottom"] == "insulating":
    problem.add_equation("Tz(z=0) = 0")
    boundary_conditions += "; Insulating bottom"
elif args["--bottom"] == "vanishing":
    problem.add_equation("Tempk(z=0) = 0")
    boundary_conditions += "; Vanishing bottom"
elif args["--bottom"] == "fixed_flux":
    problem.add_equation("Tz(z=0) = -F")
    boundary_conditions += "; Fixed flux bottom"
else:
    raise ValueError(
        f'Invalid bottom boundary condition {args["--bottom"]}, must be "insulating", "vanishing" or "fixed_flux"'
    )


#! === Other ===
# * === Currie et al. 2020 ===
# # Fixed temp bottom, insulating top:
# problem.add_equation("Temp(z=0) = 0")
# problem.add_equation("Tz(z=Lz) = 0")

# ? === Velocity Boundary Conditions ===
# * === Stress-Free ===
# d(ux)/dz|(z=0, D) = 0
if args["--slip"] == "no":
    # * === No-Slip  ===
    problem.add_equation("u(z=0) = 0")
    problem.add_equation("u(z=Lz) = 0")
    boundary_conditions += "; No-slip"
elif args["--slip"] == "free":
    # * === Free-Slip ===
    problem.add_equation("dzu_y(z=0) = 0")
    problem.add_equation("dzu_y(z=Lz) = 0")
    problem.add_equation("dzu_x(z=0) = 0")
    problem.add_equation("dzu_x(z=Lz) = 0")
    problem.add_equation("u_z(z=0) = 0")
    problem.add_equation("u_z(z=Lz) = 0")
    boundary_conditions += "; Free-Slip"
else:
    raise ValueError(
        f"invalid slip condition {args['--slip']}, must be 'no' or 'free'."
    )

# Pressure gauge fixing
problem.add_equation("integ(p) = 0")

logger.info("Boundary conditions: {}".format(boundary_conditions))

solver = problem.build_solver(timestepper)
logger.info("Solver built")

# ====================
# INITIAL CONDITIONS
# ====================
if args["--input"]:
    if pathlib.Path(restart_path + "snapshots/").exists():
        restart_file = sorted(glob(restart_path + "snapshots/*.h5"))[-1]
        write, last_dt = solver.load_state(restart_file, -1)
        dt = last_dt
        first_iter = solver.iteration
        fh_mode = "append"
    else:
        logger.error(
            "Problem reading file.\n{} does not exist.".format(
                restart_path + "snapshots_s1.h5"
            )
        )
        exit(-10)
else:
    Temp.fill_random("g", seed=42, distribution="normal", scale=1e-5)
    # Temp.low_pass_filter(scales=0.25)
    # Temp.high_pass_filter(scales=0.125)
    Temp["g"] *= z * (Lz - z)
    Temp["g"] += Lz - z

    first_iter = 0
    dt = max_timestep
    fh_mode = "overwrite"

if not args["--test"]:
    os.makedirs(outpath + "run_params/", exist_ok=True)
    run_params = {
        "Ly": Ly,
        "Lz": Lz,
        "Ny": Ny,
        "Nz": Nz,
        "Ra": Ra,
        "Pr": Pr,
        "Ta": Ta,
        "theta": theta_deg,
        "F": F,
        "max_timestep": max_timestep,
        "snapshot_iter": snapshot_iter,
        "slices_iter": slices_iter,
        "horiz_iter": horiz_iter,
        "scalar_iter": scalar_iter,
    }
    run_params = json.dumps(run_params, indent=4)

    with open(outpath + "run_params/runparams.json", "w") as run_file:
        run_file.write(run_params)

    # ====================
    #     3D DATA FIELD
    # ====================
    snapshots = solver.evaluator.add_file_handler(
        outpath + "snapshots",
        iter=snapshot_iter,
        max_writes=5000,
        mode=fh_mode,
        parallel=parallel,
    )
    snapshots.add_tasks(solver.state, layout="g")
    # ==================
    #       SLICES
    # ==================
    slices = solver.evaluator.add_file_handler(
        outpath + "slices",
        iter=slices_iter,
        max_writes=5000,
        mode=fh_mode,
        parallel=parallel,
    )
    slices.add_task(Temp(x=0), name="T(x=0)", layout="g")
    slices.add_task(u(x=0), name="u(x=0)", layout="g")
    slices.add_task(Temp(x=0.5 * Ly), name="T(x=0.5)", layout="g")
    slices.add_task(u(x=0.5 * Ly), name="u(x=0.5)", layout="g")
    slices.add_task(Temp(z=Ly), name="T(x=1)", layout="g")
    slices.add_task(u(z=Ly), name="u(x=1)", layout="g")
    slices.add_task(Temp(z=Lz / 2), name="T(z=0.5)", layout="g")
    slices.add_task(u(z=Lz / 2), name="u(z=0.5)", layout="g")

    # ==================
    #   HORIZONTAL AVE
    # ==================
    horiz_aves = solver.evaluator.add_file_handler(
        outpath + "horiz_aves",
        iter=horiz_iter,
        max_writes=2500,
        mode=fh_mode,
        parallel=parallel,
    )
    horiz_aves.add_task(
        d3.Integrate(d3.Integrate(Temp, "x"), "y") / (Ly * Ly), name="<T>", layout="g"
    )
    horiz_aves.add_task(
        d3.Integrate(d3.Integrate(f_cond, "x"), "y") / (Ly * Ly),
        name="<F_cond>",
        layout="g",
    )
    horiz_aves.add_task(
        d3.Integrate(d3.Integrate(f_conv, "x"), "y") / (Ly * Ly),
        name="<F_conv>",
        layout="g",
    )

    # ==================
    #      SCALARS
    # ==================
    scalars = solver.evaluator.add_file_handler(
        outpath + "scalars",
        iter=scalar_iter,
        max_writes=5000,
        mode=fh_mode,
        parallel=parallel,
    )
    scalars.add_task(
        d3.Integrate(d3.Integrate(d3.Integrate(0.5 * u @ u, "y"), "z"), "x")
        / (Lz * Ly * Ly),
        name="KE",
        layout="g",
    )
    scalars.add_task(
        d3.Integrate(d3.Integrate(d3.Integrate(np.sqrt(u @ u), "x"), "y"), "z")
        / (Lz * Ly * Ly),
        name="Re",
        layout="g",
    )
    scalars.add_task(
        d3.Integrate(d3.Integrate(Temp(z=0), "y"), "x") / (Ly * Ly),
        name="<T(0)>",
        layout="g",
    )
    scalars.add_task(
        d3.Integrate(d3.Integrate(d3.Integrate(Temp, "x"), "y"), "z") / (Ly * Ly * Lz),
        name="<<T>>",
        layout="g",
    )
    scalars.add_task(
        d3.Integrate(d3.Integrate(d3.Integrate(f_cond + f_conv, "x"), "y"), "z")
        / (Ly * Ly * Lz),
        name="F_tot",
        layout="g",
    )

    # analysis = solver.evaluator.add_file_handler(
    #     outpath + "analysis",
    #     iter=analysis_iter,
    #     max_writes=5000,
    #     mode=fh_mode,
    #     parallel=parallel,
    # )
    # analysis.add_task(f_cond, name='F_cond', layout='g') #? F_cond
    # analysis.add_task(f_conv, name='F_conv', layout='g') #? F_conv
    # analysis.add_task(0.5*u@u, name='KE', layout='g') #? KE
    # analysis.add_task(d3.Integrate(Temp, 'y') / Ly, name='<T>y', layout='g') #? <T>y
    # analysis.add_task(d3.Integrate(d3.Integrate(Temp, 'y'), 'z') / (Lz*Ly), name='<T>', layout='g') #? <T>
    # analysis.add_task((d3.Integrate(f_cond, coords['z']) / Lz) / (d3.Integrate(f_conv, coords['z']) / Lz),
    #                   name='Nu_inst', layout='g') #? Nu_inst

solver.stop_sim_time = stop_sim_time
# solver.stop_wall_time = stop_wall_time
# solver.stop_iteration = first_iter + rp.end_iteration + 1
solver.warmup_iterations = solver.iteration + 2000

CFL = d3.CFL(
    solver,
    initial_dt=dt,
    cadence=10,
    safety=0.5,
    threshold=0.1,
    max_change=1.5,
    min_change=0.5,
    max_dt=max_timestep,
)
CFL.add_velocity(u)
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(np.sqrt(u @ u), name="Re")
if args["--kill"]:
    exit(-99)
try:
    logger.info("Starting main loop")
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration - 1) % 10 == 0:
            max_Re = flow.max("Re")
            logger.info(
                "Iteration=%i,\n\tTime=%e, dt=%e, max(Re)=%f"
                % (solver.iteration - first_iter, solver.sim_time, timestep, max_Re)
            )
        if np.isnan(max_Re) or np.isinf(max_Re):
            raise NaNFlowError
except KeyboardInterrupt:
    logger.error("User quit loop. Triggering end of main loop")
    exit_code = -1
except NaNFlowError:
    logger.error("Max Re is NaN or inf. Triggering end of loop")
    exit_code = -50
except:
    logger.error("Unknown error raised. Triggering end of loop")
    exit_code = -10
finally:
    # if not args.test:
    #     # logger.info("Merging outputs...")
    #     # combine_outputs.merge_files(outpath)
    solver.evaluate_handlers_now(timestep)
    solver.log_stats()
    total_iterations = solver.iteration - first_iter
    snap_writes = (total_iterations) // snapshot_iter
    horiz_writes = (total_iterations) // horiz_iter
    scalar_writes = (total_iterations) // scalar_iter
    logger.info(
        "Snaps = {}, Horiz = {}, Scalars = {}".format(
            snap_writes, horiz_writes, scalar_writes
        )
    )
    logger.info("Written to {}".format(outpath))
    exit(exit_code)
