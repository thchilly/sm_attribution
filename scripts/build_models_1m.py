#!/usr/bin/env python3
from __future__ import annotations
import argparse
from sm_attribution.io.registry import Registry
from sm_attribution.preprocess.harmonize_models import harmonize_all

DEFAULT_MODELS = ["h08", "hydropy", "jules-w2", "miroc-integ-land", "watergap2-2e", "web-dhm-sg"]

def main():
    ap = argparse.ArgumentParser(description="Build 0–1 m homogenized NetCDFs for selected models.")
    ap.add_argument("--registry", default="configs/data_registry.yml", help="Path to data_registry.yml")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Subset of models to process")
    args = ap.parse_args()

    reg = Registry(args.registry)
    written = harmonize_all(args.models, reg)

    print("WROTE:")
    for p in written:
        print("  ", p)

if __name__ == "__main__":
    main()