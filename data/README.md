# Data

## `data/inputs/` (in the repository)

| File | Content | Source and terms |
|---|---|---|
| `external_comparisons.json` | Cost projections, observed prices, learning-index benchmarks, Finnish forecast vintages and regional storage figures used in the comparison figures and tables | Compiled by the author from the sources cited in the paper; each entry names its source. CC BY 4.0 for the compilation. |
| `dea_2025_li_ion.json` | Values read from the Danish Energy Agency's *Technology Data for Energy Storage*, datasheet version 0011, sheet "180 Lithium Ion Battery" (May 2025), converted to EUR2025 | Danish Energy Agency, <https://ens.dk/media/6589/download>. The original workbook is not redistributed; download it from the link. |

All other inputs, with their sources and evidence classes, are in `model/params.json`.

## `data/raw/` (in the Zenodo archive, not in the repository)

The market data are 3-minute to hourly time series for 1 January 2023 to 15 November 2026.

**Fingrid open data** (`fingrid_<id>.csv`), licensed CC BY 4.0 by Fingrid Oyj, <https://data.fingrid.fi>. The Zenodo archive contains the snapshot used in the paper. `fingrid_manifest.json` lists each dataset with its name, row count and SHA-256 checksum.

| ID | Dataset |
|---|---|
| 424 | Total installed BESS capacity (MW) |
| 398, 399 | BESS discharging and charging power, real-time (3 min) |
| 124 | Electricity consumption in Finland (15 min) |
| 316, 317 | FCR-N hourly procured volume and price |
| 315, 318 | FCR-D up hourly procured volume and price |
| 281, 283 | FCR-D down hourly procured volume and price |
| 1, 2, 52, 51 | aFRR up and down procured capacity and marginal capacity price |
| 327, 329 | mFRR capacity market up volume and price (hourly) |
| 244, 106 | mFRR up- and down-regulation energy price (15 min) |
| 375, 376 | Activated mFRR balancing regulation up and down (15 min, from 4 March 2025) |
| 34, 33 | Ordered up- and down-regulations, balancing energy market (1 h, to 4 March 2025) |
| 349, 350 | aFRR energy activation volume up and down (15 min, from 12 June 2024) |
| 347, 348 | aFRR energy price VWAP up and down (15 min) |
| 371 | Electric boiler consumption, 15-minute means (`fingrid_371_15min.csv`, made by `model/downsample.py`). The 1-minute source series is not archived; the model does not use it. |

**ENTSO-E Transparency Platform** day-ahead prices for the Finnish bidding zone (`entsoe_da_FI.csv`). These are not redistributed, because day-ahead prices originate from the nominated electricity market operators and carry their terms of use. Download them with your own security token (`python model/fetch_entsoe.py --only da`). `entsoe_manifest.json` holds the SHA-256 checksum of the file used in the paper, and `run_all.py` reports whether your download matches it.

## Fetching the data yourself

```powershell
python model/fetch_fingrid.py --start 2023-01-01 --end 2026-11-15
python model/fetch_entsoe.py --only da --start 2023-01-01 --end 2026-11-15
```

Keys and tokens are read from a git-ignored `.env` file in the repository root (`FINGRID_API_KEY=...`, `ENTSOE_API_KEY=...`).
