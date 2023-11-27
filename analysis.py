"""
An analysis script for 2.5D Cartesian convection. The script can create gifs showing
a heatmap of convection, as well as time-tracks of the Kinetic Energy and 1/<T>, the
time-averaged flux balance with depth, and both the instantaneous and time-averaged
Nusselt Number.

Usage:
    analysis.py FILE [options]
    
Options:
    -t --time-tracks                # Plot the time-track data (KE and 1/<T>)
    -d --depth-profile              # Plot the depth profile of the temperature
    -f --flux-balance               # Plot the flux-balance with depth
    -i --info                       # Information required 
    -g --gif                        # Create a gif of the convection
    -h --help                       # Display this help message
    -v --version                    # Display the version
    --cadence [CADENCE]             # Cadence of the gifs [default: 1]
    --ASI [TIME]                    # Sim-time to begin average [default: 0.65]

"""
from docopt import docopt
import h5py as h5
import numpy as np
import re
import time as timer
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
import warnings
import imageio.v2 as imageio
from os import makedirs
from os.path import normpath
from shutil import rmtree
from glob import glob


def rolling_average(quantity, time, window=0.1):
    assert len(time) == len(quantity)
    run_ave = []
    for i, t0 in enumerate(time):
        mean = np.nanmean(
            quantity[(time > t0 - window / 2) & (time <= t0 + window / 2)]
        )
        run_ave.append(mean)
    return np.array(run_ave)


def get_index(time, start_time):
    return np.abs(time - start_time).argmin()


args = docopt(__doc__, version="2.0")
print(args)

direc = normpath(args["FILE"]) + "/"

if not glob(direc):
    raise FileNotFoundError("Directory does not exist")

outpath = direc + "images/"
makedirs(outpath, exist_ok=True)

start_time = timer.time()
print("====== Data Read-In ======")
if args["--info"] or args["--time-tracks"]:
    scalar_files = glob(direc + "scalars/scalars_s*.h5")
    scalar_files.sort(key=lambda f: int(re.sub("\D", "", f)))
    print(scalar_files)
    for i, sc_file in enumerate(scalar_files):
        if i == 0:
            with h5.File(sc_file, "r") as file:
                sc_time = np.array(file["scales"]["sim_time"])
                deltaT = np.array(file["tasks"]["<T(0)>"])[:, 0, 0, 0]
                if args["--time-tracks"]:
                    KE = np.array(file["tasks"]["KE"])[:, 0, 0, 0]
        else:
            with h5.File(sc_file, "r") as file:
                sc_time = np.concatenate(
                    (sc_time, np.array(file["scales"]["sim_time"])), axis=0
                )
                deltaT = np.concatenate(
                    (deltaT, np.array(file["tasks"]["<T(0)>"])[:, 0, 0, 0]), axis=0
                )
                if args["--time-tracks"]:
                    KE = np.concatenate(
                        (KE, np.array(file["tasks"]["KE"])[:, 0, 0, 0]), axis=0
                    )
    print(sc_time)
if args["--flux-balance"] or args["--depth-profile"]:
    horiz_files = glob(direc + "horiz_aves/horiz_aves_s*.h5")
    horiz_files.sort(
        key=lambda f: int(re.sub("\D", "", f))
    )  # sort by number, avoiding s1, s10, s11, etc.
    for i, h_file in enumerate(horiz_files):
        if i == 0:
            with h5.File(h_file, "r") as file:
                horiz_time = np.array(file["scales"]["sim_time"])
                z = np.array(file["tasks"]["<T>"].dims[3]["z"])
                if args["--flux-balance"]:
                    F_cond = np.array(file["tasks"]["<F_cond>"])[:, 0, 0, :]
                    F_conv = np.array(file["tasks"]["<F_conv>"])[:, 0, 0, :]
                if args["--depth-profile"]:
                    temp_hor = np.array(file["tasks"]["<T>"])[:, 0, 0, :]
        else:
            with h5.File(h_file, "r") as file:
                horiz_time = np.concatenate(
                    (horiz_time, np.array(file["scales"]["sim_time"])), axis=0
                )
                if args["--flux-balance"]:
                    F_cond = np.concatenate(
                        (F_cond, np.array(file["tasks"]["<F_cond>"])[:, 0, 0, :]),
                        axis=0,
                    )
                    F_conv = np.concatenate(
                        (F_conv, np.array(file["tasks"]["<F_conv>"])[:, 0, 0, :]),
                        axis=0,
                    )
                if args["--depth-profile"]:
                    temp_hor = np.concatenate(
                        (temp_hor, np.array(file["tasks"]["<T>"])[:, 0, 0, :]), axis=0
                    )

if args["--gif"]:
    snap_files = glob(direc + "snapshots/snapshots_s*.h5")
    snap_files.sort(key=lambda f: int(re.sub("\D", "", f)))
    for i, s_file in enumerate(snap_files):
        if i == 0:
            with h5.File(s_file, "r") as file:
                snap_time = np.array(file["scales"]["sim_time"])
                temp = np.array(file["tasks"]["Temp"])[:, 0, :, :]
                vel = np.array(file["tasks"]["u"])[:, 1:3, 0, :, :]
                y = np.array(file["tasks"]["Temp"].dims[2]["y"])
                z = np.array(file["tasks"]["Temp"].dims[3]["z"])
        else:
            with h5.File(s_file, "r") as file:
                snap_time = np.concatenate(
                    (snap_time, np.array(file["scales"]["sim_time"])), axis=0
                )
                temp = np.concatenate(
                    (temp, np.array(file["tasks"]["Temp"])[:, 0, :, :]), axis=0
                )
                vel = np.concatenate(
                    (vel, np.array(file["tasks"]["u"])[:, 1:3, 0, :, :]), axis=0
                )


read_finish = timer.time() - start_time
print(f"Done ({read_finish:.2f} seconds)")
# ! ============== Nusselt Number ============== ! #
if args["--info"]:
    print("====== Nusselt Number ======")
    Nu_start = timer.time()
    ASI = get_index(sc_time, float(args["--ASI"]))
    AEI = get_index(sc_time, float(2.0))
    AEI = None
    dT = np.nanmean(deltaT[ASI:AEI], axis=0)

    print(f"\t ΔT = {dT:.3f}\n" + f"\t 1/ΔT = {1/dT:.3f}")

    with open(direc + "run_params/runparams.json", "r") as file:
        run_params = json.load(file)
        Ra = run_params["Ra"]
        Pr = run_params["Pr"]
        Ta = run_params["Ta"]
    with open(direc + "Nu.json", "w") as file:
        json.dump(
            {"Ra": Ra, "Pr": Pr, "Ta": Ta, "ΔT": dT, "Nu": 1 / dT}, file, indent=4
        )
    print(f"Done ({timer.time() - Nu_start:.2f}s).")

# # ! ============== Time-Tracks ============== ! #
if args["--time-tracks"]:
    print("====== Time-Tracks ======")
    time_start = timer.time()
    if len(deltaT) > 50000:
        skip_cadence = 100
    elif len(deltaT) > 5000:
        skip_cadence = 10
    else:
        skip_cadence = 1
    AEI = get_index(sc_time, float(1.9))
    AEI = None
    # skip_cadence = 1

    KE_run_ave = rolling_average(KE[::skip_cadence], sc_time[::skip_cadence])
    Nu = 1 / deltaT
    Nu_run_ave = rolling_average(Nu[::skip_cadence], sc_time[::skip_cadence])
    fig, [KE_ax, Nu_ax] = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    KE_ax.plot(
        sc_time[:AEI:skip_cadence], KE_run_ave[:AEI], label="Kinetic Energy", c="r"
    )
    ylims = KE_ax.get_ylim()
    KE_ax.scatter(sc_time[:AEI:skip_cadence], KE[:AEI:skip_cadence], marker="+", c="k")
    KE_ax.set_ylim(ylims)
    KE_ax.set_xlabel(r"Time, $\tau$")
    KE_ax.set_ylabel("KE")

    Nu_ax.plot(
        sc_time[:AEI:skip_cadence], Nu_run_ave[:AEI], label="Nusselt Number", c="r"
    )
    ylims = Nu_ax.get_ylim()
    Nu_ax.scatter(sc_time[:AEI:skip_cadence], Nu[:AEI:skip_cadence], marker="+", c="k")
    Nu_ax.set_ylim(ylims)
    Nu_ax.set_xlabel(r"Time, $\tau$")
    Nu_ax.set_ylabel("Nu")
    plt.tight_layout()
    plt.savefig(outpath + "time_tracks.pdf")
    print(f"Done ({timer.time() - time_start:.2f}s).")

# # ! ============== Flux Balance ============== ! #
if args["--flux-balance"]:
    print("====== Flux Balance ======")
    flux_start = timer.time()
    #     with open(direc+'run_params/runparams.json', 'r') as file:
    #         params = json.load(file)
    #         driving_flux = params['F']
    ASI = get_index(horiz_time, float(args["--ASI"]))
    # AEI = get_index(horiz_time, float(2.0))
    AEI = None
    f_tot = F_cond + F_conv
    F_cond_bar = np.nanmean(F_cond[ASI:AEI], axis=0)
    F_conv_bar = np.nanmean(F_conv[ASI:AEI], axis=0)
    F_tot_bar = np.nanmean(f_tot[ASI:AEI], axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(F_cond_bar, z, label=r"$F_{cond}$", c="b")
    ax.plot(F_conv_bar, z, label=r"$F_{conv}$", c="r")
    ax.plot(F_tot_bar, z, label=r"$F_{tot}$", c="k")
    ax.set_xlabel("Flux")
    ax.set_ylabel("z")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(outpath + "flux_balance.pdf")

    print(f"Done ({timer.time() - flux_start:.2f}s).")

# # ! ============== Temperature Profiles ============== ! #
if args["--depth-profile"]:
    print("====== Depth Profile ======")
    depth_start = timer.time()
    ASI = get_index(horiz_time, float(args["--ASI"]))
    temp_hor_bar = np.nanmean(temp_hor[ASI:], axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(temp_hor_bar, z, c="k")
    ax.set_xlabel(r"$\langle \overline{T} \rangle$")
    ax.set_ylabel("z")
    ax.text(0.05, 0.05, f"ΔT = {temp_hor_bar[0]:.2f}", transform=ax.transAxes)
    plt.tight_layout()
    plt.savefig(outpath + "t_ave_depth_profile.pdf")
    print(f"Done ({timer.time() - depth_start:.2f}s).")
    depth_start = timer.time()
    print("Creating depth profile animation...")
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    makedirs(f"{direc}/plots", exist_ok=True)
    fnames = []
    cadence = int(args["--cadence"])
    for i, t in enumerate(horiz_time):
        if i % cadence == 0:
            print(f"\t Plotting frame {i+1}/{len(horiz_time)}", end="\r")
            ax.plot(temp_hor[i, :], z, c="k")
            ax.set_xlabel(r"$\langle \overline{T} \rangle$")
            ax.set_ylabel("z")
            ax.set_title(f"t = {t:.2f}")
            ax.text(0.05, 0.05, f"ΔT = {temp_hor[i, 0]:.2f}", transform=ax.transAxes)
            fnames.append(f"{direc}/plots/t_{i:04d}.png")
            plt.savefig(fnames[-1])
            ax.clear()
    print("\nDone. Creating GIF...")

    with imageio.get_writer(f"{outpath}/depth_profile.gif", mode="I") as writer:
        for i, filename in enumerate(fnames):
            print(f"\t frame {i+1}/{len(fnames)}", end="\r")
            image = imageio.imread(filename)
            writer.append_data(image)

    rmtree(f"{direc}/plots")
    print(f"\nDone ({timer.time() - depth_start:.2f}s).")

if args["--gif"]:
    print("====== Heatmap GIF ======")
    heatmap_start = timer.time()
    zz, yy = np.meshgrid(z, y)
    fnames = []
    vmax = 0.5
    # vmax = np.max(temp[len(temp) // 2:])
    # vmin = 1e-3
    # vmin = np.min(temp[len(temp) // 3:])
    vmin = np.min(temp)
    print(vmin, vmax)
    linthresh = np.abs(vmin)
    # cNorm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax, clip=False)

    # cNorm = mpl.colors.SymLogNorm(linthresh=0.05, linscale=1, vmin=vmin, vmax=vmax, clip=False)
    levels = np.logspace(-2, 0, 1000, endpoint=True)
    levels = np.linspace(0, 1, 1000)
    # cNorm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=False)
    # levels = np.linspace(vmin, vmax, 1000, endpoint=True)
    # all values of temp below 0, set to 0
    # temp[temp < 0] = vmin
    print("Plotting Frames...")
    makedirs(f"{direc}/plots", exist_ok=True)
    yspace = len(y) // 15
    zspace = len(z) // 5
    cadence = int(args["--cadence"])
    mpl.ticker.Locator.MAXTICKS = 1100
    for i, t in enumerate(snap_time):
        if i % cadence == 0:
            print(f"\t{(i+1) / len(snap_time) * 100:3.0f}% complete", end="\r")
            fig, ax = plt.subplots()
            # cax = fig.add_axes([0.90, 0.1, 0.02, 0.8])
            # cb1 = mpl.colorbar.ColorbarBase(cax, cmap='inferno', norm=cNorm, extend='min')
            # fig.subplots_adjust(left=0.1, right=0.85)
            # print(f"{i+1}/{len(snap_time)}", end='\r')
            # cax = ax.contourf(yy, zz, temp[i, :, :], cmap='inferno', extend='min', levels=levels, norm=cNorm)
            cax = ax.contourf(
                yy, zz, temp[i, :, :], cmap="inferno", extend="min", levels=999
            )
            warnings.filterwarnings("ignore")
            ax.quiver(
                yy[::yspace, ::zspace],
                zz[::yspace, ::zspace],
                vel[i, 0, ::yspace, ::zspace],
                vel[i, 1, ::yspace, ::zspace],
                color="w",
                pivot="mid",
            )
            ax.set_xlabel("y")
            ax.set_ylabel("z")
            ax.set_title(rf"{t:.2f} $\tau$")
            fig.colorbar(cax, label=r"$T$")
            fnames.append(f"{direc}plots/{i:04d}.png")
            plt.savefig(fnames[-1])
            fig.clf()
    print("\nCreating GIF...")
    with imageio.get_writer(f"{outpath}/heatmap.gif", mode="I") as writer:
        for i, filename in enumerate(fnames):
            print(f"\t frame {i+1}/{len(fnames)}", end="\r")
            image = imageio.imread(filename)
            writer.append_data(image)
    rmtree(f"{direc}/plots")
    print(f"\nDone ({timer.time() - heatmap_start:.2f}s).")
