# src/sm_attribution/metadata/obs_groups.py
"""
Forcing-based grouping of observational soil moisture datasets.

Groups:
    1 = Reanalysis-based
    2 = Satellite-based
    3 = Hybrid (Reanalysis + Satellite)
    4 = In-situ / ML-based
"""

OBS_GROUPS = {
    # Reanalysis-based
    "era5-land": {
        "group": 1,
        "description": (
            "ERA5-Land: land surface model (HTESSEL) driven by ERA5 atmospheric "
            "reanalysis; no satellite forcings — fully reanalysis-based."
        ),
    },
    "gleam-42a": {
        "group": 1,
        "description": (
            "GLEAM v4.2a: core meteorological forcings (radiation, temperature, "
            "vapor pressure deficit, wind) from MSWX reanalysis; precipitation "
            "from MSWEP (gauge + reanalysis + satellite). Dominantly reanalysis-driven."
        ),
    },
    "merra2-land": {
        "group": 1,
        "description": (
            "MERRA-2 Land: NASA Catchment LSM forced by the MERRA-2 "
            "reanalysis atmosphere; includes DA but fundamentally reanalysis-driven."
        ),
    },
    "gdo-smia": {
        "group": 1,
        "description": (
            "GDO-SMIA: LISFLOOD hydrological model driven by ECMWF/EFAS "
            "reanalysis meteorology; no satellite inputs. Purely reanalysis-based."
        ),
    },

    # Satellite-based
    "gleam-42b": {
        "group": 2,
        "description": (
            "GLEAM v4.2b: uses satellite-only forcings — CERES radiation, "
            "AIRS temperature, IMERG precipitation, ESA CCI SM. Satellite-driven version."
        ),
    },
    "grace-da-dm": {
        "group": 2,
        "description": (
            "GRACE-DA-DM: GRACE satellite gravity anomalies assimilated into "
            "the Catchment LSM. Root-zone signal ultimately derived from satellite data."
        ),
    },

    # Hybrid
    "gldas-v21": {
        "group": 3,
        "description": (
            "GLDAS v2.1: forced by GDAS reanalysis, GPCP (gauges + satellite), "
            "and AGRMET radiation. Strong mix of reanalysis and EO forcings."
        ),
    },
    "gdo-ensmia": {
        "group": 3,
        "description": (
            "GDO-ENSMIA: LISFLOOD model (reanalysis-forced) combined with "
            "MODIS LST and ESA CCI SM (satellite). Clear hybrid product."
        ),
    },

    # In-situ / ML
    "somo-ml": {
        "group": 4,
        "description": (
            "SoMo.ml: machine learning model trained primarily on ISMN in-situ "
            "soil moisture observations with auxiliary static features. Not satellite or reanalysis driven."
        ),
    },
}