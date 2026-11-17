# Battery energy storage in Finland to 2050

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24322467.svg)](https://doi.org/10.5281/zenodo.24322467)

Model, data and paper for **Battery energy storage in Finland to 2050: a multi-method forecast of power, energy, cost and electricity use** (Merilehto, 2026). The study estimates Finland's grid-connected battery fleet in GW and GWh, its duration, its cost per kWh and per MWh delivered, and its own electricity use, for 2027–2050. It is part of the *Electric markets 2050* series.

Every model-derived number in the paper is written by the model run into `paper/generated/`, so the text, tables and figures cannot drift from the code.

## Results

Pooled planning states (Low, Base and High are the 10th, 50th and 90th percentiles of a five-method linear opinion pool):

| Year | Power, GW: Low / **Base** / High | Energy, GWh: Low / **Base** / High | Fleet h |
|---|---|---|---|
| 2030 | 4.0 / **4.7** / 6.1 | 9.2 / **11.8** / 16.4 | 2.5 |
| 2035 | 3.6 / **7.0** / 11.9 | 8.3 / **21.8** / 40.6 | 2.7 |
| 2040 | 3.6 / **7.7** / 17.7 | 7.8 / **25.2** / 60.8 | 2.9 |
| 2050 | 4.8 / **10.2** / 21.8 | 12.6 / **34.2** / 79.2 | 3.3 |

- New systems shift from two to four hours in the mid-2030s; eight-hour systems rarely pay.
- The fleet's own revenue erosion and the growth of flexibility need explain most of the variance of the economic method, followed by the cost of capital and grid connection. Global technology learning matters little.
- Conversion losses and auxiliary consumption draw about 1 % of national electricity consumption in 2050.

## Repository layout

```
model/            model code, parameter set (params.json) and tests
  run_all.py      runs every step in order and writes outputs/manifest.json
  cost.py         component cost model (cells, DC BOS, PCS, European premium)
  spreads.py      perfect-foresight day-ahead arbitrage
  reserves.py     reserve and balancing revenue pools
  f_equilibrium.py, f_methods.py, ensemble.py, stress.py
                  the five forecast methods, pooling, Sobol and stress cases
  make_numbers.py writes paper/generated/numbers.tex and the generated tables
  fetch_fingrid.py, fetch_entsoe.py
                  data download (your own API key and token)
figures/          figure scripts and rendered figures
outputs/          derived data written by the model
data/inputs/      compiled external figures (forecasts, cost benchmarks)
data/raw/         market data (not in this repository; see data/README.md)
paper/            LaTeX source, bibliography, generated values and the PDF
```

## Reproduce

Requires Python 3.11 or later with NumPy, SciPy and Matplotlib, and a LaTeX distribution with `latexmk` for the paper. In PowerShell:

```powershell
conda env create -f environment.yml
conda activate bess-fi-2050
python model/run_all.py
python -m unittest discover -s model/tests
latexmk -cd -pdf paper/bess_finland_2050.tex
```

Without `data/raw/`, the run rebuilds the forecast, tables and figures from the derived market series in `outputs/`. To rerun every step from the raw market data, either unpack the Zenodo archive, which contains the Fingrid snapshot, or download the data yourself:

1. Put your keys in a `.env` file in the repository root. It is git-ignored; never commit it.
   ```
   FINGRID_API_KEY=<key from https://developer-data.fingrid.fi>
   ENTSOE_API_KEY=<token from https://transparency.entsoe.eu>
   ```
2. Download the data, then rerun:
   ```powershell
   python model/fetch_fingrid.py          # not needed with the Zenodo archive
   python model/fetch_entsoe.py --only da
   python model/run_all.py
   ```

`outputs/manifest.json` lists which raw files matched the SHA-256 checksums of the snapshot used in the paper. Fingrid and ENTSO-E occasionally revise past data, so a fresh download can differ slightly. The run is deterministic given the seed in `params.json`.

## Citation

> Merilehto, J. (2026). *Battery energy storage in Finland to 2050: a multi-method forecast of power, energy, cost and electricity use* (Version 1.0.0). Zenodo. https://doi.org/10.5281/zenodo.24322467

See `CITATION.cff` for machine-readable metadata.

## Licence

Code (`*.py`): MIT, see `LICENSE`. Paper, figures, parameter set and derived data: CC BY 4.0, see `LICENSE-CC-BY-4.0.md`. Third-party data keep their own terms, see `data/README.md`.

The analysis and views expressed are solely those of the author and do not represent any employer or affiliated institution.
