#!/usr/bin/env python3
"""
Stream-download + crop + regrid TROPOMI (Sentinel-5P L2 NO2) files.

Behavior:
 - Reads a list of s3:// URLs from urls.txt (one per line)
 - Downloads first file (temp file)
 - While processing that file, starts downloading next file concurrently
 - Processes: subset to bbox, apply qa mask, regrid to regular lon/lat grid
 - Writes processed (small) NetCDF in OUTPUT_DIR and deletes the raw temp file
"""

import os
import subprocess
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import numpy as np
import xarray as xr
from pyresample import geometry, kd_tree

# ====== CONFIG ======
URLS_FILE = "sp5_no2_download_urls.txt"   # one s3://... URL per line
TEMP_DIR = "./tmp_s5p"                    # temporary raw swath storage (local)
OUTPUT_DIR = "/Users/eliaserv/Documents/VSCode_projects/SCEA/SCEA/data/sentinel-5p/S5P_NO2_June2024_regrid"  # processed outputs (small files)
S3CMD_CONFIG = ".s3cfg"                   # s3cmd config (in current dir) or full path
BBOX = [-10.0, 30.0, 45.0, 72.0]          # [min_lon, min_lat, max_lon, max_lat] - Europe
GRID_RES = 0.05                           # degrees for output grid (change as needed, e.g. 0.01)
QA_THRESHOLD = 0.70                       # if qa_value exists, mask < this
RETRY_DOWNLOAD = 5                        # number of retries per file
# ====================

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_s3(url, out_path, s3cfg=None):
    """
    Download an s3:// URL to out_path using s3cmd.
    Raise subprocess.CalledProcessError on failure.
    """
    cmd = ["s3cmd"]
    if s3cfg:
        cmd += ["-c", s3cfg]
    cmd += ["get", "--continue", url, out_path]
    # run and raise on error
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

def find_no2_var(ds: xr.Dataset):
    """Find the most likely NO2 data variable name inside the dataset."""
    # common variable names include tropospheric_NO2_column_number_density or NO2_column_number_density
    candidates = [v for v in ds.data_vars if "no2" in v.lower()]
    if len(candidates) == 0:
        raise KeyError("No NO2 variable found in dataset. Available vars: " + ", ".join(ds.data_vars.keys()))
    # pick the most descriptive (longest name) or first
    return sorted(candidates, key=lambda s: -len(s))[0]

def process_and_save(input_nc, out_nc, bbox=BBOX, res_deg=GRID_RES, qa_threshold=QA_THRESHOLD):
    """
    1) open input netcdf (swath)
    2) select only points inside bbox (to save memory)
    3) apply QA mask if available
    4) resample/regrid to regular lat/lon grid using pyresample (nearest + radius_of_influence)
    5) save as small netcdf with coords lon/lat
    """
    print(f"[proc] Opening {input_nc}")
    ds = xr.open_dataset(input_nc, mask_and_scale=True)

    # locate lat/lon variables (common names: latitude, longitude)
    # TROPOMI L2 typically uses 'latitude' and 'longitude' 2D arrays
    lat_names = [n for n in ds.coords.keys() if "lat" in n.lower()]
    lon_names = [n for n in ds.coords.keys() if "lon" in n.lower()]
    if not lat_names or not lon_names:
        # sometimes lat/lon are data_vars
        lat_names = [n for n in ds.data_vars.keys() if "lat" in n.lower()]
        lon_names = [n for n in ds.data_vars.keys() if "lon" in n.lower()]
    if not lat_names or not lon_names:
        raise KeyError("Could not find latitude/longitude variables in dataset")

    lat = ds[lat_names[0]].values
    lon = ds[lon_names[0]].values

    # find the NO2 variable
    varname = find_no2_var(ds)
    da = ds[varname]
    print(f"[proc] Found NO2 variable: {varname}")

    # If qa_value exists, apply mask
    if "qa_value" in ds:
        qa = ds["qa_value"].values
        mask = np.where(qa >= qa_threshold, 1, 0)
        da_vals = np.where(mask, da.values, np.nan)
    else:
        da_vals = da.values

    # Quick bbox mask: keep points whose lon/lat fall into bbox
    min_lon, min_lat, max_lon, max_lat = bbox
    inside_mask = (lon >= min_lon) & (lon <= max_lon) & (lat >= min_lat) & (lat <= max_lat)
    if not inside_mask.any():
        print("[proc] WARNING: no points found inside bbox for this file.")
        # still create empty grid (all nan)
    # Create source swath def (pyresample expects 1D or 2D arrays)
    swath_def = geometry.SwathDefinition(lons=lon, lats=lat)

    # Build target grid (regular lon/lat)
    # lon increases east, lat increases north. We'll create lons as ascending and lats descending to match common raster ordering
    target_lons = np.arange(min_lon + res_deg/2.0, max_lon + res_deg/2.0, res_deg)
    target_lats = np.arange(max_lat - res_deg/2.0, min_lat - res_deg/2.0, -res_deg)  # descending
    lons2d, lats2d = np.meshgrid(target_lons, target_lats)
    grid_def = geometry.GridDefinition(lons=lons2d, lats=lats2d)

    # Use kd_tree.resample_nearest (fast and robust). radius_of_influence in meters; TROPOMI swath pixels ~3.5x7 km
    # radius_of_influence can be tuned (e.g. 25000 to 50000 m)
    radius_of_influence = 40000  # meters
    print("[proc] Performing resampling (this may take a few seconds)...")
    result = kd_tree.resample_nearest(swath_def, da_vals, grid_def,
                                     radius_of_influence=radius_of_influence,
                                     fill_value=np.nan)

    # Construct xarray Dataset for output
    out_lat_1d = (target_lats)  # descending
    out_lon_1d = (target_lons)
    # create DataArray with dims ('y','x')
    da_out = xr.DataArray(result, dims=("y", "x"),
                          coords={"lat": (("y","x"), lats2d),
                                  "lon": (("y","x"), lons2d)},
                          attrs={
                              "long_name": da.attrs.get("long_name", varname),
                              "units": da.attrs.get("units", "")
                          })

    ds_out = xr.Dataset({varname: da_out})
    # add 1D coords if you prefer
    ds_out = ds_out.assign_coords({"lat_1d": (("y",), out_lat_1d), "lon_1d": (("x",), out_lon_1d)})

    # metadata
    ds_out.attrs["source_file"] = os.path.basename(input_nc)
    ds_out.attrs["bbox"] = ",".join(map(str, bbox))
    ds_out.attrs["grid_res_deg"] = res_deg

    print(f"[proc] Writing output -> {out_nc}")
    encoding = {varname: {"zlib": True, "complevel": 4}}
    ds_out.to_netcdf(out_nc, encoding=encoding)
    ds.close()
    ds_out.close()
    print(f"[proc] Saved {out_nc}")


def run_pipeline(urls, s3cfg=S3CMD_CONFIG):
    """
    Main pipeline: overlap download/process. Assumes urls is a list.
    """
    total = len(urls)
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    def temp_name(i):
        return os.path.join(TEMP_DIR, f"tmp_{i:04d}.nc")

    def out_name(i, src_url):
        # create a deterministic output name (date+index or use source filename)
        base = os.path.basename(src_url)
        # create short processed name:
        return os.path.join(OUTPUT_DIR, f"proc_{i:04d}_{base}")

    with ThreadPoolExecutor(max_workers=2) as ex:
        # start first download
        futures = {}
        if total == 0:
            print("No URLs to process.")
            return

        # start download of first
        i = 0
        tmp0 = temp_name(i)
        print(f"[dl] Scheduling download 0/{total-1}")
        futures[i] = ex.submit(_download_with_retries, urls[i], tmp0, s3cfg)

        pbar = tqdm(total=total, desc="processed files")
        while i < total:
            # wait for current download to finish (blocking)
            future = futures[i]
            try:
                future.result()  # raises if download failed
            except Exception as e:
                print(f"[dl] Download failed for index {i}, url={urls[i]}: {e}")
                # skip to next file (start next if exists)
                # schedule next:
                i += 1
                if i < total:
                    tmp_next = temp_name(i)
                    futures[i] = ex.submit(_download_with_retries, urls[i], tmp_next, s3cfg)
                continue

            # schedule next download (so it overlaps with processing)
            next_idx = i + 1
            if next_idx < total:
                tmp_next = temp_name(next_idx)
                futures[next_idx] = ex.submit(_download_with_retries, urls[next_idx], tmp_next, s3cfg)

            # process current
            tmpfile = temp_name(i)
            out_file = out_name(i, urls[i])
            try:
                process_and_save(tmpfile, out_file)
            except Exception as e:
                print(f"[proc] ERROR processing {tmpfile}: {e}")
            finally:
                # delete tmp file if it exists to free space
                if os.path.exists(tmpfile):
                    try:
                        os.remove(tmpfile)
                    except Exception as e:
                        print(f"[cleanup] Failed to delete {tmpfile}: {e}")

            pbar.update(1)
            i += 1

        pbar.close()

def _download_with_retries(url, tmp_path, s3cfg, retries=RETRY_DOWNLOAD):
    last_exc = None
    for attempt in range(1, retries+1):
        try:
            # ensure temp dir exists
            os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            print(f"[dl] Downloading (attempt {attempt}) {url} -> {tmp_path}")
            download_s3(url, tmp_path, s3cfg=s3cfg)
            print(f"[dl] Completed {url}")
            return True
        except subprocess.CalledProcessError as e:
            last_exc = e
            print(f"[dl] attempt {attempt} failed for {url}: {e}")
            time.sleep(2 * attempt)
    raise last_exc

if __name__ == "__main__":
    # read urls
    with open(URLS_FILE) as f:
        urls_list = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]

    print(f"Found {len(urls_list)} urls.")
    run_pipeline(urls_list, s3cfg=S3CMD_CONFIG)
    print("All done.")