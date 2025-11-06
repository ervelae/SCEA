# Get list of S5P NO2 data files over Europe for June 2024
# using the Copernicus Data Space Ecosystem STAC API

from pystac_client import Client
from tqdm import tqdm

client = Client.open("https://stac.dataspace.copernicus.eu/v1/")


TIME_RANGE = "2024-06-01/2024-06-30"
# approx. Europe bbox [min_lon, min_lat, max_lon, max_lat]
BBOX = [-10.0, 30.0, 45.0, 72.0]
TXT_OUT_FILE = "/Users/eliaserv/Documents/VSCode_projects/SCEA/SCEA/data/sentinel-5p/sp5_no2_download_urls.txt"
DOWNLOAD_DIRECTORY = "S5P_NO2_June2024"
COLLECTION = "no2-offl"


get_file_txt = False
if get_file_txt:

    all_collections = client.get_all_collections()

    # Seach for COLLECTION in all_collections
    for col in all_collections:
        if COLLECTION in col.id:
            print(f"[+] Found collection {col} \t collection_id = {col.id} \t title={col.title}")
            found_collection = col
            break

    print(f"[+] Searching collection {found_collection.id} for {TIME_RANGE} bbox={BBOX} ...")
    search = client.search(collections=[found_collection.id], datetime=TIME_RANGE, bbox=BBOX, limit=100)
    items_gen = search.item_collection()
    urls = set()  # python sets dont allow for duplicates and do not maintain order
    count = 0
    for item in tqdm(items_gen, desc="Items", unit="item"):
        count += 1
        # prefer asset named "data" or assets ending with .nc
        for k, asset in item.assets.items():
            href = asset.href
            # common checks for netCDF / .nc assets
            if href and (href.lower().endswith(".nc") or ".nc?" in href.lower() or "netcdf" in (asset.get("type","") or "").lower() or "netcdf" in (asset.get("media_type","") or "").lower()):
                urls.add(href)
                break
            else:
                print(f"[-] Skipping asset {k} with href={href} type={asset.get('type','')} media_type={asset.get('media_type','')}")

    print(f"[+] Processed {count} items, found {len(urls)} candidate download URLs.")

    with open(TXT_OUT_FILE, "w") as fh:
        for u in sorted(urls):
            fh.write(u + "\n")
    print(f"[+] Wrote {len(urls)} URLs to {TXT_OUT_FILE}")



download_intructions = rf"""
INSTRUCTIONS:
1. set up config file for s3cmd

cd {TXT_OUT_FILE}

nano .s3cfg

    [default]
    access_key = <access_key>
    host_base = eodata.dataspace.copernicus.eu
    host_bucket = eodata.dataspace.copernicus.eu
    human_readable_sizes = False
    secret_key = <secret_key>
    use_https = true
    check_ssl_certificate = true

2. Download data

FILE="{TXT_OUT_FILE}"
TOTAL=$(wc -l < "$FILE")
count=0

while IFS= read -r url; do
  if s3cmd -c .s3cfg get "$url" {DOWNLOAD_DIRECTORY} --continue; then
    count=$((count+1))
  else
    echo "download failed: $url" >&2
  fi
  # simple progress: "3/42"
  printf ' \r%d/%d ' "$count" "$TOTAL" >&2
done < "$FILE"
printf '\n' >&2
"""

print(
download_intructions
    )