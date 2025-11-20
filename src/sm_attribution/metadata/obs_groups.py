# src/sm_attribution/metadata/obs_groups.py
"""
Forcing-based grouping of observational soil moisture datasets,
including plotting settings (color and marker).

Groups:
    1 = Reanalysis-based
    2 = Satellite-based
    3 = Hybrid (Reanalysis + Satellite)
    4 = In-situ / ML-based
"""

# ---------------------------------------------------------------------
# Group-level plotting settings
# ---------------------------------------------------------------------

# Marker shapes per group (used for group legend)
GROUP_MARKERS = {
    1: "o",  # Reanalysis-based
    2: "s",  # Satellite-based
    3: "^",  # Hybrid
    4: "D",  # In-situ / ML-based
}

# Human-readable group labels (for legends, titles, etc.)
GROUP_LABELS = {
    1: "Reanalysis-based",
    2: "Satellite-based",
    3: "Hybrid (Rean.+Sat.)",
    4: "In-situ / ML-based",
}

# ---------------------------------------------------------------------
# Dataset-level plotting settings
# ---------------------------------------------------------------------

# Distinct colours for each observational dataset
OBS_COLORS = {
    "era5-land":   "#1b9e77",
    "gleam-42a":   "#d95f02",
    "gleam-42b":   "#7570b3",
    "gldas-v21":   "#e7298a",
    "somo-ml":     "#66a61e",
    "merra2-land": "#e6ab02",
    "grace-da-dm": "#a6761d",
    "gdo-ensmia":  "#666666",
    "gdo-smia":    "#1f78b4",
}

# ---------------------------------------------------------------------
# Full metadata per observational dataset
# ---------------------------------------------------------------------

OBS_GROUPS = {
    # Reanalysis-based
    "era5-land": {
        "group": 1,
        "marker": GROUP_MARKERS[1],
        "color": OBS_COLORS["era5-land"],
        "description": (
            "ERA5-Land: land surface model (HTESSEL) driven by ERA5 atmospheric "
            "reanalysis; no satellite forcings — fully reanalysis-based."
        ),
    },
    "gleam-42a": {
        "group": 1,
        "marker": GROUP_MARKERS[1],
        "color": OBS_COLORS["gleam-42a"],
        "description": (
            "GLEAM v4.2a: core meteorological forcings (radiation, temperature, "
            "vapor pressure deficit, wind) from MSWX reanalysis; precipitation "
            "from MSWEP (gauge + reanalysis + satellite). Dominantly reanalysis-driven."
        ),
    },
    "merra2-land": {
        "group": 1,
        "marker": GROUP_MARKERS[1],
        "color": OBS_COLORS["merra2-land"],
        "description": (
            "MERRA-2 Land: NASA Catchment LSM forced by the MERRA-2 "
            "reanalysis atmosphere; includes DA but fundamentally reanalysis-driven."
        ),
    },
    "gdo-smia": {
        "group": 1,
        "marker": GROUP_MARKERS[1],
        "color": OBS_COLORS["gdo-smia"],
        "description": (
            "GDO-SMIA: LISFLOOD hydrological model driven by ECMWF/EFAS "
            "reanalysis meteorology; no satellite inputs. Purely reanalysis-based."
        ),
    },

    # Satellite-based
    "gleam-42b": {
        "group": 2,
        "marker": GROUP_MARKERS[2],
        "color": OBS_COLORS["gleam-42b"],
        "description": (
            "GLEAM v4.2b: uses satellite-only forcings — CERES radiation, "
            "AIRS temperature, IMERG precipitation, ESA CCI SM. Satellite-driven version."
        ),
    },
    "grace-da-dm": {
        "group": 2,
        "marker": GROUP_MARKERS[2],
        "color": OBS_COLORS["grace-da-dm"],
        "description": (
            "GRACE-DA-DM: GRACE satellite gravity anomalies assimilated into "
            "the Catchment LSM. Root-zone signal ultimately derived from satellite data."
        ),
    },

    # Hybrid
    "gldas-v21": {
        "group": 3,
        "marker": GROUP_MARKERS[3],
        "color": OBS_COLORS["gldas-v21"],
        "description": (
            "GLDAS v2.1: forced by GDAS reanalysis, GPCP (gauges + satellite), "
            "and AGRMET radiation. Strong mix of reanalysis and EO forcings."
        ),
    },
    "gdo-ensmia": {
        "group": 3,
        "marker": GROUP_MARKERS[3],
        "color": OBS_COLORS["gdo-ensmia"],
        "description": (
            "GDO-ENSMIA: LISFLOOD model (reanalysis-forced) combined with "
            "MODIS LST and ESA CCI SM (satellite). Clear hybrid product."
        ),
    },

    # In-situ / ML
    "somo-ml": {
        "group": 4,
        "marker": GROUP_MARKERS[4],
        "color": OBS_COLORS["somo-ml"],
        "description": (
            "SoMo.ml: machine learning model trained primarily on ISMN in-situ "
            "soil moisture observations with auxiliary static features. Not satellite or reanalysis driven."
        ),
    },
}