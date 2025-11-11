# scripts/batch_make_ssi.py
import argparse
from sm_attribution.io.registry import default_registry
from sm_attribution.analysis.ensemble import ensure_all_models, ensure_all_obs

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["standalone","pooled"], default="standalone")
parser.add_argument("--scale", type=int, default=3)
parser.add_argument("--ref-start", default="2003-01")
parser.add_argument("--ref-end", default="2019-12")
args = parser.parse_args()

MODE = args.mode
SCALE = args.scale
REF_START = args.ref_start
REF_END = args.ref_end

MODELS = ["h08","hydropy","jules-w2","miroc-integ-land","watergap2-2e","web-dhm-sg","lpjml5-7-10-fire"]
OBS = [
    "era5land_1950_2020","gleam42a_1980_2020","gleam42a_2003_2020","gleam42b_2003_2020",
    "gldas_v21_2000_2020","somo_ml_0p5m_2000_2019","gracedadm_2003_2020","merra2_1980_2020",
]

def main():
    reg = default_registry()
    model_paths = ensure_all_models(MODELS, reg.scenarios(), reg=reg, scale=SCALE, ref_start=REF_START, ref_end=REF_END, mode=MODE)
    obs_paths   = ensure_all_obs(OBS, reg=reg, scale=SCALE, ref_start=REF_START, ref_end=REF_END)
    print(f"Models SSI (mode={MODE}, ref window={REF_START} to {REF_END}):")
    for (m,s), p in model_paths.items():
        print(f"  {m:18s} {s:20s} -> {p}")
    print("\nObs SSI:")
    for k, p in obs_paths.items():
        print(f"  {k:25s} -> {p}")

if __name__ == "__main__":
    main()