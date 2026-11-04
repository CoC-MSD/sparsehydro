"""
SWMM network travel-time and production-run utilities for swmmrdii.

Functions
---------
compute_network_lags(inp_path, outlet_node, sc_df, n_jobs, sim_days)
    Estimate per-SC travel time (junction → gage) via parallel SWMM pulse
    tests using the openswmm engine.  Returns {sc_name: lag_days}.

prepare_swmm_run(model, cal, inp_path, artifacts_dir, sim_start, sim_end)
    Generate per-SC RDII .tsf files and a patched SWMM INP ready to open in
    PCSWMM.  Returns the path to the patched INP.

patch_inp(inp_path, out_path, replacements)
    Copy a SWMM INP file, replacing or appending named sections.

rescale_imperv(bundle, new_imperv_pct, new_mean_perv)
    Adjust ia_max in a loaded pkl bundle for a new %imperviousness value
    (future-condition scenario analysis).

rescale_bfi(bundle, new_bfi, new_mean_bfi)
    Adjust R_3 in a loaded pkl bundle for a new BFI value (donor-bundle
    transfer to unmetered subcatchments, or revised BFI estimates).
"""

from __future__ import annotations

import copy
import json
import math
import os
import pickle
import re
import tempfile
import time
from typing import Optional

import pandas as pd

from .io import SH_SERIES_PREFIX


# ---------------------------------------------------------------------------
# INP section parsing helpers
# ---------------------------------------------------------------------------

# Conduit-graph cache keyed by (abspath, mtime_ns, size) — see
# _parse_conduit_graph.  The graphs are tiny (one entry per link);
# the parse they avoid is a scan of a potentially huge INP.
_GRAPH_CACHE: dict[tuple, tuple[dict, dict]] = {}
_GRAPH_CACHE_MAX = 4


def _parse_inp_sections(inp_path: str) -> dict[str, str]:
    """
    Read a SWMM INP file and return a dict of section_name → raw_content.

    Section names include the brackets, e.g. ``'[JUNCTIONS]'``.
    The content is the text of that section *without* the section header line.
    """
    sections: dict[str, str] = {}
    current: Optional[str] = None
    buf: list[str] = []

    with open(inp_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                if current is not None:
                    sections[current] = "".join(buf)
                current = stripped.upper()
                buf = []
            else:
                buf.append(line)

    if current is not None:
        sections[current] = "".join(buf)

    return sections


def _merge_section(existing: str, new_rows: str, key_col: int = 0) -> str:
    """
    Merge *new_rows* into an existing INP section body.

    Rows in *new_rows* whose first whitespace-delimited token matches an
    existing row's token are replaced; rows with new tokens are appended.
    Comment lines (starting with ``;;``) from *new_rows* are only written
    once (de-duplicated against the existing header).

    Parameters
    ----------
    existing : str
        Current section body (without the ``[SECTION]`` header line).
    new_rows : str
        Replacement/addition rows (without the ``[SECTION]`` header line).
    key_col : int
        Column index (0-based) used to identify a row's key token.
    """
    def _key(line: str) -> Optional[str]:
        parts = line.split()
        if not parts or parts[0].startswith(";;"):
            return None
        return parts[key_col].upper() if len(parts) > key_col else None

    # Build ordered dict of existing rows keyed by first token.
    existing_lines: list[str] = existing.splitlines(keepends=True)
    rows: dict[str, str] = {}     # key -> line (preserves order via insertion)
    ordered_keys: list[str] = []  # insertion order
    for ln in existing_lines:
        k = _key(ln)
        if k is None:
            ordered_keys.append(f"\x00comment\x00{id(ln)}")
            rows[ordered_keys[-1]] = ln
        elif k not in rows:
            rows[k] = ln
            ordered_keys.append(k)
        else:
            rows[k] = ln  # overwrite duplicate key in existing (keep last)

    # Apply new rows: replace existing keys or append new ones.
    for ln in new_rows.splitlines(keepends=True):
        k = _key(ln)
        if k is None:
            # Skip comment/header lines that are already present.
            stripped = ln.strip()
            if not any(v.strip() == stripped for v in rows.values()):
                tag = f"\x00comment\x00{id(ln)}"
                rows[tag] = ln
                ordered_keys.append(tag)
        elif k in rows:
            rows[k] = ln   # replace existing row with matching key
        else:
            rows[k] = ln
            ordered_keys.append(k)

    return "".join(rows[k] for k in ordered_keys)


def patch_inp(
    inp_path: str,
    out_path: str,
    replacements: dict[str, str],
    merge_sections: Optional[set] = None,
) -> None:
    """
    Copy a SWMM INP, replacing or merging named sections.

    Parameters
    ----------
    inp_path : str
        Source INP file (unmodified).
    out_path : str
        Destination path for the patched INP.
    replacements : dict
        Mapping of section name (e.g. ``'[INFLOWS]'``) → new section content
        (the text that follows the header line, not including the header).
        Sections not in *replacements* are copied verbatim.  Sections in
        *replacements* that do not exist in the source are appended.
    merge_sections : set or None
        Section names (e.g. ``{'[INFLOWS]', '[TIMESERIES]'}``) that should
        be *merged* rather than wholesale replaced.  Merge keeps existing
        rows whose key token does not appear in the new content, then adds
        or replaces rows from the new content.  Use this to preserve
        pre-existing DWF or other inflow entries alongside sparsehydro rows.
        When ``None`` (default), all sections in *replacements* are replaced.
    """
    norm = {k.strip().upper(): v for k, v in replacements.items()}
    merge = {s.strip().upper() for s in (merge_sections or set())}
    sections = _parse_inp_sections(inp_path)
    # Capture the source section order BEFORE applying replacements — avoids
    # re-parsing the file (expensive for PCSWMM exports with embedded
    # [TIMESERIES]) just to recover ordering.
    original_keys = list(sections.keys())

    for key, content in norm.items():
        if key in merge and key in sections:
            sections[key] = _merge_section(sections[key], content)
        else:
            sections[key] = content
    written: set[str] = set()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for key in original_keys:
            f.write(f"{key}\n")
            f.write(sections.get(key, ""))
            written.add(key)
        for key in sections:
            if key not in written:
                f.write(f"\n{key}\n")
                f.write(sections[key])


# ---------------------------------------------------------------------------
# Stripped (routing-only) INP builder
# ---------------------------------------------------------------------------

# Hydrology processes neutralized in the sparsehydro-driven run.  Sections
# (subcatchments, rain gages, hydrographs, ...) are intentionally KEPT so the
# binary output's entity list matches the source model — PCSWMM refuses to
# bind a .out whose subcatchment/node/link sets differ from the open model.
_IGNORE_OPTIONS = (
    ("IGNORE_RAINFALL", "YES"),
    ("IGNORE_SNOWMELT", "YES"),
    ("IGNORE_GROUNDWATER", "YES"),
    ("IGNORE_RDII", "YES"),
)


def _build_routing_only_inp(
    inp_path: str,
    out_path: str,
    sim_start: str,
    sim_end: str,
    sh_nodes: Optional[set] = None,
    files_content: Optional[str] = None,
) -> None:
    """
    Write an INP for a sparsehydro-driven (routing-only) SWMM run.

    Keeps the full entity set so the .out matches the source model.  Native
    hydrology is disabled via IGNORE_* options rather than deleting sections.

    [INFLOWS] handling: rows referencing a time series named ``SH_*`` are
    removed — those are a previous sparsehydro static export, and the
    interface file re-supplies the same flows (double-count).  ALL other
    rows are preserved: they are external boundary forcings the model owner
    set up deliberately (point sources, a reservoir's net precip/evap
    series, interbasin transfers) and SWMM applies them additively
    alongside the ``USE INFLOWS`` interface file — including at nodes that
    also receive sparsehydro inflows (e.g. a storage node fed by both
    tributary RDII and its own surface net precip/evap).  Rows kept at
    *sh_nodes* (case-insensitive) are listed in a notice so the user can
    confirm they are genuinely additive.  With ``sh_nodes=None`` the whole
    section is cleared (legacy behavior).
    """
    sections = _parse_inp_sections(inp_path)

    if sh_nodes is None:
        # Legacy behavior: sparsehydro supplies all inflows.
        sections.pop("[INFLOWS]", None)
    elif "[INFLOWS]" in sections:
        sh_lower = {str(n).lower() for n in sh_nodes}
        kept, dropped, kept_at_sh = [], [], []
        for line in sections["[INFLOWS]"].splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith(";"):
                kept.append(line)
                continue
            tokens = stripped.split()
            series = tokens[2] if len(tokens) >= 3 else ""
            if series.upper().startswith(SH_SERIES_PREFIX.upper()):
                dropped.append(line)
                continue
            kept.append(line)
            if tokens[0].lower() in sh_lower:
                kept_at_sh.append(tokens[0])
        kept_nodes = [
            ln.strip().split()[0] for ln in kept
            if ln.strip() and not ln.strip().startswith(";")
        ]
        if kept_nodes:
            sections["[INFLOWS]"] = "\n".join(kept) + "\n"
            msg = f"[INFLOWS]   : kept {len(kept_nodes)} external inflow row(s)"
            if dropped:
                msg += f"; removed {len(dropped)} stale SH_* export row(s)"
            print(msg)
            if kept_at_sh:
                print(
                    f"              note: {', '.join(sorted(set(kept_at_sh))[:5])}"
                    f"{', ...' if len(set(kept_at_sh)) > 5 else ''} also "
                    f"receive sparsehydro inflows - SWMM adds both "
                    f"(verify the [INFLOWS] series is a separate source, "
                    f"not subcatchment runoff)"
                )
        else:
            sections.pop("[INFLOWS]", None)

    # Patch OPTIONS: simulation window + disable native hydrology.
    opts = sections.get("[OPTIONS]", "")
    opts = re.sub(r"(?m)^(START_DATE\s+).*$", rf"\g<1>{sim_start}", opts)
    opts = re.sub(r"(?m)^(END_DATE\s+).*$", rf"\g<1>{sim_end}", opts)
    opts = re.sub(r"(?m)^(REPORT_START_DATE\s+).*$", rf"\g<1>{sim_start}", opts)
    if "START_DATE" not in opts:
        opts += f"START_DATE            {sim_start}\n"
    if "END_DATE" not in opts:
        opts += f"END_DATE              {sim_end}\n"
    for opt, val in _IGNORE_OPTIONS:
        if re.search(rf"(?m)^{opt}\b", opts):
            opts = re.sub(rf"(?m)^({opt}\s+).*$", rf"\g<1>{val}", opts)
        else:
            opts += f"{opt}       {val}\n"
    sections["[OPTIONS]"] = opts

    # [FILES] injection (e.g. 'USE INFLOWS "..."') — done here rather than a
    # separate patch_inp() pass so the (potentially huge) INP is read and
    # written exactly once.
    if files_content is not None:
        sections["[FILES]"] = files_content

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for key, content in sections.items():
            f.write(f"{key}\n{content}")


# ---------------------------------------------------------------------------
# Topological network lag computation (conduit length / velocity)
# ---------------------------------------------------------------------------

def _parse_conduit_graph(inp_path: str) -> tuple[dict, dict]:
    """
    Parse [CONDUITS], [XSECTIONS], and [JUNCTIONS] from *inp_path*, plus the
    hydraulic link sections ([WEIRS], [ORIFICES], [PUMPS], [OUTLETS]) as
    zero-length edges — a weir or pump connects two nodes with no physical
    travel length, but without the edge a subcatchment routing through such
    a node (e.g. a reservoir discharging over a weir) has NO path to the
    outlet and would silently receive lag 0.

    Returns
    -------
    graph : dict[str, list[tuple[str, str, float]]]
        Adjacency list: ``{from_node: [(to_node, conduit_name, length_ft), ...]}``.
    conduit_props : dict[str, dict]
        Per-conduit properties used for travel-time estimation:
        ``{conduit_name: {"shape": str, "geom1": float, "manning_n": float,
                          "slope": float}}``.
        ``slope`` is the dimensionless bed slope (ft/ft), computed from
        junction invert elevations + conduit offsets.  ``shape`` is
        upper-cased.  IRREGULAR sections have slope computed but Manning's
        is not applied (no analytical hydraulic radius available).
    """
    from .swmm_parse import scan_swmm_sections

    # Cached by (path, mtime, size): the graph is re-requested within one
    # notebook cell (compute_network_lags for the SCs, then bc_node_lag),
    # and each uncached parse costs a scan of the INP — hundreds of MB for
    # PCSWMM exports with embedded [TIMESERIES].  Copies are returned so
    # callers can never mutate the cached master.
    st = os.stat(inp_path)
    cache_key = (os.path.abspath(inp_path), st.st_mtime_ns, st.st_size)
    cached = _GRAPH_CACHE.get(cache_key)
    if cached is not None:
        graph, conduit_props = cached
        return (
            {k: list(v) for k, v in graph.items()},
            {k: dict(v) for k, v in conduit_props.items()},
        )

    sections = scan_swmm_sections(
        inp_path,
        ("CONDUITS", "XSECTIONS", "JUNCTIONS",
         "WEIRS", "ORIFICES", "PUMPS", "OUTLETS"),
    )
    conduit_rows = sections["CONDUITS"]
    xsec_rows    = sections["XSECTIONS"]
    junc_rows    = sections["JUNCTIONS"]

    # JUNCTIONS: Name  InvertElev  MaxDepth  InitDepth  SurDepth  Aponded
    junc_invert: dict[str, float] = {}
    for row in junc_rows:
        if len(row) < 2:
            continue
        try:
            junc_invert[row[0]] = float(row[1])
        except ValueError:
            pass

    # CONDUITS: Name  FromNode  ToNode  Length  ManningN  InOffset  OutOffset  InitFlow  MaxFlow
    graph: dict[str, list] = {}
    conduit_meta: dict[str, dict] = {}
    for row in conduit_rows:
        if len(row) < 4:
            continue
        name, from_n, to_n = row[0], row[1], row[2]
        try:
            length = float(row[3])
        except ValueError:
            continue
        manning_n = float(row[4]) if len(row) > 4 else 0.013
        in_off    = float(row[5]) if len(row) > 5 else 0.0
        out_off   = float(row[6]) if len(row) > 6 else 0.0
        graph.setdefault(from_n, []).append((to_n, name, length))
        conduit_meta[name] = {
            "from_n": from_n, "to_n": to_n,
            "length": length, "manning_n": manning_n,
            "in_off": in_off, "out_off": out_off,
        }

    # XSECTIONS: Link  Shape  Geom1  Geom2  Geom3  Geom4  Barrels  Culvert
    # For IRREGULAR, Geom1 column holds the transect name instead of a number.
    xsec: dict[str, dict] = {}
    for row in xsec_rows:
        if len(row) < 3:
            continue
        cname = row[0]
        shape = row[1].upper()
        if shape == "IRREGULAR":
            xsec[cname] = {"shape": shape, "geom1": 0.0, "transect": row[2]}
        else:
            try:
                geom1 = float(row[2])
            except ValueError:
                geom1 = 0.0
            geom2 = float(row[3]) if len(row) > 3 else 0.0
            geom3 = float(row[4]) if len(row) > 4 else 0.0
            geom4 = float(row[5]) if len(row) > 5 else 0.0
            xsec[cname] = {"shape": shape, "geom1": geom1, "geom2": geom2,
                           "geom3": geom3, "geom4": geom4, "transect": None}

    # Assemble conduit_props: merge xsec + slope computed from junction inverts.
    conduit_props: dict[str, dict] = {}
    for cname, meta in conduit_meta.items():
        xs    = xsec.get(cname, {})
        shape = xs.get("shape", "IRREGULAR")
        geom1 = xs.get("geom1", 0.0)
        el_from = junc_invert.get(meta["from_n"])
        el_to   = junc_invert.get(meta["to_n"])
        if el_from is not None and el_to is not None and meta["length"] > 0:
            # slope = (upstream invert + in_offset - downstream invert - out_offset) / length
            slope = ((el_from + meta["in_off"]) - (el_to + meta["out_off"])) / meta["length"]
        else:
            slope = 0.0
        conduit_props[cname] = {
            "shape":     shape,
            "geom1":     geom1,
            "geom2":     xs.get("geom2", 0.0),
            "geom3":     xs.get("geom3", 0.0),
            "geom4":     xs.get("geom4", 0.0),
            "transect":  xs.get("transect"),   # transect name for IRREGULAR, else None
            "manning_n": meta["manning_n"],
            "slope":     slope,
        }

    # Hydraulic links: zero-length edges so lag paths continue through
    # storage/diversion structures.  Travel time is 0 by construction
    # (length 0), for both the constant and Manning branches.
    for section in ("WEIRS", "ORIFICES", "PUMPS", "OUTLETS"):
        for row in sections[section]:
            if len(row) < 3:
                continue
            name, from_n, to_n = row[0], row[1], row[2]
            graph.setdefault(from_n, []).append((to_n, name, 0.0))
            conduit_props[name] = {
                "shape": "HYDRAULIC_LINK", "geom1": 0.0, "geom2": 0.0,
                "geom3": 0.0, "geom4": 0.0, "transect": None,
                "manning_n": 0.013, "slope": 0.0,
            }

    if len(_GRAPH_CACHE) >= _GRAPH_CACHE_MAX:
        _GRAPH_CACHE.pop(next(iter(_GRAPH_CACHE)))
    _GRAPH_CACHE[cache_key] = (graph, conduit_props)
    return (
        {k: list(v) for k, v in graph.items()},
        {k: dict(v) for k, v in conduit_props.items()},
    )


def _parse_transects(inp_path: str) -> dict[str, str]:
    """
    Parse the ``[TRANSECTS]`` section and return a dict of transect name → raw block text.

    Each transect consists of one ``NC`` line followed by one ``X1`` line and
    one or more ``GR`` lines.  The transect name is the first token of the
    ``X1`` line.  The returned block text includes all lines for that transect
    (NC + X1 + GR…) and can be written verbatim into a branch INP.
    """
    sections = _parse_inp_sections(inp_path)
    raw = sections.get("[TRANSECTS]", "")

    transects: dict[str, str] = {}
    current_name: Optional[str] = None
    current_lines: list[str] = []

    for line in raw.splitlines(keepends=True):
        stripped = line.strip()
        if not stripped or stripped.startswith(";"):
            continue
        tag = stripped.split()[0].upper()
        if tag == "NC":
            # NC always precedes the X1 it belongs to — buffer it
            current_lines = [line]
        elif tag == "X1":
            # First token after X1 is the transect name
            parts = stripped.split()
            if len(parts) >= 2:
                if current_name is not None:
                    transects[current_name] = "".join(current_lines)
                current_name = parts[1]
                current_lines.append(line)
        elif tag == "GR":
            current_lines.append(line)

    if current_name is not None:
        transects[current_name] = "".join(current_lines)

    return transects


def _conduit_travel_time_d(
    conduit_name: str,
    length_ft: float,
    conduit_props: dict,
    velocity_fps: float,
) -> float:
    """
    Return travel time in days for one conduit segment.

    If *velocity_fps* > 0, uses that constant velocity for all conduits.

    If *velocity_fps* == 0, attempts Manning's equation using per-conduit
    slope (from junction invert elevations) and Manning's n (from [CONDUITS]):

        v = (1.486 / n) * R^(2/3) * S^(1/2)   [US customary, ft/s]

    Hydraulic radius R is estimated at half-full depth, which is the
    typical design point and gives a conservative (slightly low) velocity:

        CIRCULAR          R = D / 4
        RECT_CLOSED       R = (W * D/2) / (W + D)      (W = geom2 or geom1)
        RECT_OPEN         R = (W * D/2) / (W + D/2)    (open top)
        TRAPEZOIDAL       R ≈ A / P  at half-depth
        PARABOLIC         R ≈ 0.5 * geom1 / 3          (approximation)
        IRREGULAR / other → falls back to constant velocity_fps=2.0

    Zero or negative slope also falls back to the constant-velocity default.
    """
    props = conduit_props.get(conduit_name, {})

    if velocity_fps > 0:
        v = velocity_fps
    else:
        shape     = props.get("shape", "IRREGULAR")
        geom1     = props.get("geom1", 0.0) or 0.0   # full height / diameter (ft)
        manning_n = props.get("manning_n", 0.013) or 0.013
        slope     = props.get("slope", 0.0)

        r = None  # hydraulic radius at half-full depth

        if slope > 0 and geom1 > 0 and shape not in ("IRREGULAR", "CUSTOM"):
            if shape == "CIRCULAR":
                r = geom1 / 4.0

            elif shape in ("RECT_CLOSED", "RECT_OPEN", "RECTANGULAR"):
                # geom1 = full height, geom2 = width (if available, else assume square)
                geom2 = props.get("geom2", geom1) or geom1
                d_half = geom1 / 2.0
                a = geom2 * d_half
                if shape == "RECT_OPEN":
                    p = geom2 + d_half        # open top: no top width in wetted perimeter
                else:
                    p = 2 * d_half + 2 * geom2
                r = a / p if p > 0 else None

            elif shape == "TRAPEZOIDAL":
                # geom1=full depth, geom2=bottom width, geom3=left slope, geom4=right slope
                geom2 = props.get("geom2", 0.0) or 0.0
                geom3 = props.get("geom3", 1.0) or 1.0
                geom4 = props.get("geom4", 1.0) or 1.0
                d_half = geom1 / 2.0
                a = (geom2 + 0.5 * (geom3 + geom4) * d_half) * d_half
                p = geom2 + d_half * ((1 + geom3**2)**0.5 + (1 + geom4**2)**0.5)
                r = a / p if p > 0 else None

            elif shape == "PARABOLIC":
                # geom1=full depth, geom2=full top width; R at half-depth ≈ 2/3 * half-depth
                r = geom1 / 3.0  # conservative approximation

        if r is not None and r > 0 and slope > 0:
            v = max(0.5, (1.486 / manning_n) * (r ** (2.0 / 3.0)) * (slope ** 0.5))
        else:
            v = 2.0  # default constant velocity for IRREGULAR or missing data

    travel_s = length_ft / v
    return travel_s / 86400.0


def _trace_path(
    start_node: str,
    outlet_node: str,
    graph: dict,
    conduit_props: dict,
    velocity_fps: float,
) -> tuple[list[str], list[tuple[str, float]]]:
    """
    Dijkstra shortest-time path from *start_node* to *outlet_node*.

    Returns
    -------
    nodes : list[str]
        Ordered node names along the path, inclusive of start and outlet.
        Empty if no path exists.
    conduits : list[tuple[str, float]]
        ``[(conduit_name, length_ft), ...]`` along the path.
        Empty if no path exists or start == outlet.
    """
    if start_node == outlet_node:
        return [start_node], []

    import heapq

    # (cost_days, node, path_nodes, path_conduits)
    heap = [(0.0, start_node, [start_node], [])]
    visited: set[str] = set()

    while heap:
        cost, node, path_nodes, path_conduits = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        if node == outlet_node:
            return path_nodes, path_conduits
        for (to_n, cname, length_ft) in graph.get(node, []):
            if to_n not in visited:
                tt = _conduit_travel_time_d(cname, length_ft, conduit_props, velocity_fps)
                heapq.heappush(heap, (
                    cost + tt, to_n,
                    path_nodes + [to_n],
                    path_conduits + [(cname, length_ft)],
                ))

    return [], []  # no path found


def _shortest_path_lag(
    start_node: str,
    outlet_node: str,
    graph: dict,
    conduit_props: dict,
    velocity_fps: float,
) -> float:
    """
    Find the minimum-travel-time path from *start_node* to *outlet_node*.
    Returns lag in days, or 0.0 if no path exists.
    """
    nodes, conduits = _trace_path(start_node, outlet_node, graph, conduit_props, velocity_fps)
    if not conduits:
        return 0.0
    return sum(
        _conduit_travel_time_d(cname, length_ft, conduit_props, velocity_fps)
        for cname, length_ft in conduits
    )


def _build_branch_inp(
    path_nodes: list[str],
    path_conduits: list[tuple[str, float]],
    outlet_node: str,
    conduit_props: dict,
    conduit_meta: dict,
    junc_invert: dict,
    transect_blocks: dict,
    pulse_cfs: float = 10.0,
    sim_hours: int = 72,
    dt_min: int = 5,
) -> str:
    """
    Build a minimal SWMM INP string for a single branch (path_nodes + path_conduits).

    The start node receives a unit pulse (triangular hydrograph peaking at
    *pulse_cfs* at t=1h, back to zero by t=2h).  The outlet node is replaced
    with a FREE outfall.  All intermediate nodes are JUNCTION.

    Parameters
    ----------
    path_nodes : list[str]
        Ordered node names from upstream junction to outlet.
    path_conduits : list[tuple[str, float]]
        ``[(conduit_name, length_ft), ...]`` in the same order.
    conduit_meta : dict
        Raw conduit metadata from ``_parse_conduit_graph`` internals.
        Must have keys: from_n, to_n, length, manning_n, in_off, out_off.
    junc_invert : dict[str, float]
        Junction invert elevations from the full model.
    transect_blocks : dict[str, str]
        Mapping of transect name → raw HEC-2 block text from ``_parse_transects``.
        Used to write ``[TRANSECTS]`` for IRREGULAR conduits.  May be empty.
    """
    lines: list[str] = []

    # OPTIONS
    from datetime import datetime, timedelta
    sim_end_dt = datetime(2000, 1, 1) + timedelta(hours=sim_hours)
    lines += [
        "[TITLE]",
        "Branch pulse test",
        "",
        "[OPTIONS]",
        "FLOW_UNITS        CFS",
        "INFILTRATION      GREEN_AMPT",
        "FLOW_ROUTING      DYNWAVE",
        "LINK_OFFSETS      DEPTH",
        "MIN_SLOPE         0",
        "ALLOW_PONDING     NO",
        "SKIP_STEADY_STATE NO",
        "",
        "START_DATE        01/01/2000",
        "START_TIME        00:00:00",
        f"END_DATE          {sim_end_dt.strftime('%m/%d/%Y')}",
        f"END_TIME          {sim_end_dt.strftime('%H:%M:%S')}",
        "REPORT_START_DATE 01/01/2000",
        "REPORT_START_TIME 00:00:00",
        f"SWEEP_START       01/01",
        f"SWEEP_END         12/31",
        "DRY_DAYS          0",
        f"REPORT_STEP       00:{dt_min:02d}:00",
        f"WET_STEP          00:{dt_min:02d}:00",
        f"DRY_STEP          00:{dt_min:02d}:00",
        f"ROUTING_STEP      00:{dt_min:02d}:00",
        "LENGTHENING_STEP  0",
        "VARIABLE_STEP     0",
        "MINIMUM_STEP      0.5",
        "INERTIAL_DAMPING  PARTIAL",
        "NORMAL_FLOW_LIMITED BOTH",
        "FORCE_MAIN_EQUATION H-W",
        "SURCHARGE_METHOD  EXTRAN",
        "MAX_TRIALS        8",
        "HEAD_TOLERANCE    0.005",
        "SYS_FLOW_TOL      5",
        "LAT_FLOW_TOL      5",
        "MINIMUM_SURFAREA  12.566",
        "THREADS           1",
        "",
    ]

    # JUNCTIONS — all nodes except the outlet
    lines.append("[JUNCTIONS]")
    lines.append(";;Name           InvertElev  MaxDepth  InitDepth  SurDepth  Aponded")
    for node in path_nodes[:-1]:
        inv = junc_invert.get(node, 0.0)
        lines.append(f"{node:<16s} {inv:>10.3f}  100       0          0         0")
    lines.append("")

    # OUTFALLS — outlet node as FREE outfall
    lines.append("[OUTFALLS]")
    lines.append(";;Name           Elevation  Type    StageData  Gated    RouteTo")
    inv = junc_invert.get(outlet_node, 0.0)
    lines.append(f"{outlet_node:<16s} {inv:>9.3f}  FREE                 NO")
    lines.append("")

    # CONDUITS
    lines.append("[CONDUITS]")
    lines.append(";;Name           FromNode         ToNode           Length     Roughness  InOffset   OutOffset  InitFlow   MaxFlow")
    for cname, _ in path_conduits:
        m = conduit_meta.get(cname, {})
        fn = m.get("from_n", "?")
        tn = m.get("to_n", "?")
        ln = m.get("length", 100.0)
        nn = m.get("manning_n", 0.013)
        io = m.get("in_off", 0.0)
        oo = m.get("out_off", 0.0)
        lines.append(
            f"{cname:<16s} {fn:<16s} {tn:<16s} {ln:>10.3f} {nn:>10.4f} {io:>10.3f} {oo:>10.3f} 0          0"
        )
    lines.append("")

    # XSECTIONS — write real geometry; for IRREGULAR use transect name from conduit_props.
    # Fall back to CIRCULAR 3 ft only if the transect is missing from transect_blocks.
    lines.append("[XSECTIONS]")
    lines.append(";;Link           Shape        Geom1      Geom2      Geom3      Geom4      Barrels    Culvert")
    needed_transects: list[str] = []
    for cname, _ in path_conduits:
        props = conduit_props.get(cname, {})
        shape = props.get("shape", "IRREGULAR")
        if shape == "IRREGULAR":
            tname = props.get("transect")
            if tname and tname in transect_blocks:
                lines.append(f"{cname:<16s} IRREGULAR    {tname}")
                needed_transects.append(tname)
            else:
                # Transect not found — fall back to a generic circular pipe
                lines.append(f"{cname:<16s} CIRCULAR      3.000 0          0          0          1          0")
        else:
            geom1 = props.get("geom1", 0.0)
            geom2 = props.get("geom2", 0.0)
            geom3 = props.get("geom3", 0.0)
            geom4 = props.get("geom4", 0.0)
            lines.append(f"{cname:<16s} {shape:<12s} {geom1:>10.3f} {geom2:>10.3f} {geom3:>10.3f} {geom4:>10.3f} 1          0")
    lines.append("")

    # TRANSECTS — include only the transects referenced by this branch
    if needed_transects:
        lines.append("[TRANSECTS]")
        lines.append(";;Transect Data in HEC-2 format")
        lines.append(";")
        for tname in needed_transects:
            lines.append(transect_blocks[tname].rstrip())
        lines.append("")

    # TIMESERIES — triangular pulse: 0 at t=0, peak at t=1h, 0 at t=2h
    ts_name = "PULSE"
    lines.append("[TIMESERIES]")
    lines.append(";;Name           Date       Time       Value")
    lines.append(f"{ts_name:<16s}            0:00       0")
    lines.append(f"{ts_name:<16s}            1:00       {pulse_cfs}")
    lines.append(f"{ts_name:<16s}            2:00       0")
    lines.append(f"{ts_name:<16s}            {sim_hours}:00      0")
    lines.append("")

    # INFLOWS — inject pulse at the start (most upstream) node
    start_node = path_nodes[0]
    lines.append("[INFLOWS]")
    lines.append(";;Node           Constituent      Time Series      Type     Mfactor  Sfactor  Baseline Pattern")
    lines.append(f"{start_node:<16s} FLOW             {ts_name:<16s} FLOW     1.0      1.0      0")
    lines.append("")

    # REPORT
    lines.append("[REPORT]")
    lines.append("SUBCATCHMENTS NONE")
    lines.append("NODES ALL")
    lines.append("LINKS ALL")
    lines.append("")

    lines.append("[END]")
    lines.append("")

    return "\n".join(lines)


def _pulse_lag_days(
    path_nodes: list[str],
    path_conduits: list[tuple[str, float]],
    outlet_node: str,
    conduit_props: dict,
    conduit_meta: dict,
    junc_invert: dict,
    transect_blocks: dict,
    fallback_days: float,
) -> float:
    """
    Run a branch-isolated SWMM pulse test and return centroid travel-time lag.

    Falls back to *fallback_days* (the constant-velocity estimate) if the
    branch cannot be modelled or SWMM errors/produces no flow — every
    fallback prints one line naming the branch and the reason, so a run
    where the pulse test silently degraded is visible in the lag output.
    """
    label = f"{path_nodes[0]} -> {outlet_node}" if path_nodes else outlet_node

    def _fb(reason: str) -> float:
        print(f"    [pulse -> constant-velocity fallback] {label}: {reason}")
        return fallback_days

    try:
        from openswmm.solver import run_solver
        from openswmm.output import Output, NodeAttribute
    except ImportError:
        return _fb("openswmm is not importable")

    # Paths that cross the zero-length hydraulic-link edges (weirs/orifices/
    # pumps/outlets) have no real conduit geometry for a branch model —
    # the generated INP would be invalid and fail in run_solver below, so
    # bail out with a clear reason instead.
    hyd = [name for name, *_ in path_conduits
           if conduit_props.get(name, {}).get("shape") == "HYDRAULIC_LINK"]
    if hyd:
        return _fb(
            f"path crosses hydraulic link(s) {hyd[:3]} "
            f"(weir/orifice/pump/outlet) with no branch-model geometry"
        )

    try:
        inp_text = _build_branch_inp(
            path_nodes, path_conduits, outlet_node,
            conduit_props, conduit_meta, junc_invert,
            transect_blocks,
        )
    except Exception as e:
        return _fb(f"branch INP build failed: {e!r}")

    with tempfile.TemporaryDirectory() as tmp:
        inp_path = os.path.join(tmp, "branch.inp")
        rpt_path = os.path.join(tmp, "branch.rpt")
        out_path = os.path.join(tmp, "branch.out")

        with open(inp_path, "w", encoding="utf-8") as f:
            f.write(inp_text)

        try:
            err = run_solver(inp_path, rpt_path, out_path)
            if err:
                return _fb(f"SWMM branch run failed (error code {err})")
        except Exception as e:
            return _fb(f"SWMM branch run raised: {e!r}")

        # Read total inflow at the outlet node from the binary .out file.
        # get_node_timeseries returns {datetime: flow_value}.
        try:
            with Output(out_path) as out:
                ts = out.get_node_timeseries(outlet_node, NodeAttribute.TOTAL_INFLOW)
        except Exception as e:
            return _fb(f"could not read branch .out: {e!r}")

    # ts keys are datetime objects; sim starts 2000-01-01 00:00
    from datetime import datetime as _dt
    t0 = _dt(2000, 1, 1)
    t_arr = [(t - t0).total_seconds() / 86400.0 for t in ts.keys()]
    q_arr = list(ts.values())

    total = sum(q_arr)
    if total <= 0:
        return _fb("no flow reached the outlet in the pulse test")

    # Flow-weighted centroid time (days from sim start)
    centroid = sum(t * q for t, q in zip(t_arr, q_arr)) / total
    # Subtract the pulse centroid time (1h = 1/24 d) to get net travel time
    net_lag = max(0.0, centroid - 1.0 / 24.0)
    return net_lag


def compute_network_lags(
    inp_path: str,
    outlet_node: str,
    sc_df: pd.DataFrame,
    method: str = "constant",
    velocity_fps: float = 2.0,
    save_path: Optional[str] = None,
    # Legacy parameters — kept for backward compatibility, no longer used.
    n_jobs: int = -1,  # noqa: ARG001
    sim_days: int = 30,  # noqa: ARG001
    sim_start: str = "01/01/2000",  # noqa: ARG001
) -> dict[str, float]:
    """
    Estimate per-SC network travel-time lags via conduit topology.

    Three methods are available, selected by *method*:

    ``"constant"`` (default)
        Length / constant velocity.  Fast, no SWMM required.  Appropriate
        for daily-timestep calibration where sub-hour lag precision is not
        needed.

    ``"manning"``
        Manning's equation for regular cross-sections (CIRCULAR, RECT_CLOSED,
        RECT_OPEN, TRAPEZOIDAL, PARABOLIC) at half-full depth, using slope
        computed from junction invert elevations.  IRREGULAR sections fall
        back to *velocity_fps* (default 2.0 fps).  Still deterministic; no
        SWMM required.

    ``"pulse"``
        Branch-isolated SWMM pulse test.  For each unique outlet junction,
        the Dijkstra path to *outlet_node* is extracted, a minimal INP is
        generated containing only those nodes and conduits, a triangular
        pulse is injected at the upstream end, and the flow-weighted centroid
        arrival time at the outlet is returned as the lag.  IRREGULAR sections
        are replaced with a 3-ft circular pipe in the branch model (the
        cross-section only affects hydraulic accuracy, not the measurement
        concept).  Falls back to ``"constant"`` lag if SWMM errors.  Requires
        openswmm to be installed.

    Parameters
    ----------
    inp_path : str
        Path to the SWMM INP file.
    outlet_node : str
        SWMM node name at the calibration gage (e.g. ``'LW_33'``).
    sc_df : pd.DataFrame
        Subcatchment table.  Must have columns ``'name'`` and ``'outlet'``.
    method : str
        Lag estimation method: ``"constant"``, ``"manning"``, or ``"pulse"``.
    velocity_fps : float
        Constant velocity fallback in ft/s (default 2.0).  Used directly by
        ``"constant"``; used as fallback by ``"manning"`` and ``"pulse"``.
    save_path : str or None
        If provided, write the lag dict as JSON to this path.
    n_jobs, sim_days, sim_start : ignored
        Kept for backward compatibility with the former pulse-test API.

    Returns
    -------
    dict
        Mapping of subcatchment name → lag in days (float).
    """
    method = method.lower()
    if method not in ("constant", "manning", "pulse"):
        raise ValueError(f"method must be 'constant', 'manning', or 'pulse'; got {method!r}")

    # velocity passed to graph traversal: for "manning" pass 0 to trigger Manning's branch
    trav_velocity = 0.0 if method == "manning" else velocity_fps

    graph, conduit_props = _parse_conduit_graph(inp_path)

    # For pulse method we also need raw conduit_meta, junc_invert, and transects.
    if method == "pulse":
        from .swmm_parse import parse_swmm_section
        _crows = parse_swmm_section(inp_path, "CONDUITS")
        _jrows = parse_swmm_section(inp_path, "JUNCTIONS")
        conduit_meta: dict[str, dict] = {}
        for row in _crows:
            if len(row) < 4:
                continue
            conduit_meta[row[0]] = {
                "from_n": row[1], "to_n": row[2],
                "length": float(row[3]),
                "manning_n": float(row[4]) if len(row) > 4 else 0.013,
                "in_off":   float(row[5]) if len(row) > 5 else 0.0,
                "out_off":  float(row[6]) if len(row) > 6 else 0.0,
            }
        junc_invert: dict[str, float] = {}
        for row in _jrows:
            if len(row) >= 2:
                try:
                    junc_invert[row[0]] = float(row[1])
                except ValueError:
                    pass
        transect_blocks = _parse_transects(inp_path)

    # Deduplicate: only one path search per unique outlet junction.
    junction_to_scs: dict[str, list[str]] = {}
    for _, row in sc_df.iterrows():
        jn = row["outlet"]
        junction_to_scs.setdefault(jn, []).append(row["name"])

    unique_junctions = list(junction_to_scs.keys())
    print(
        f"compute_network_lags [{method}]: "
        f"{len(unique_junctions)} unique outlet junctions -> {outlet_node}"
    )

    junction_lag: dict[str, float] = {}
    for jn in unique_junctions:
        path_nodes, path_conduits = _trace_path(
            jn, outlet_node, graph, conduit_props, trav_velocity
        )

        if method == "pulse":
            if not path_conduits:
                lag = 0.0
            else:
                fallback = sum(
                    _conduit_travel_time_d(cname, length_ft, conduit_props, velocity_fps)
                    for cname, length_ft in path_conduits
                )
                lag = _pulse_lag_days(
                    path_nodes, path_conduits, outlet_node,
                    conduit_props, conduit_meta, junc_invert,
                    transect_blocks,
                    fallback_days=fallback,
                )
        else:
            lag = sum(
                _conduit_travel_time_d(cname, length_ft, conduit_props, trav_velocity)
                for cname, length_ft in path_conduits
            )

        junction_lag[jn] = lag

    # Fan lag out to all SCs sharing each junction.
    sc_lags: dict[str, float] = {}
    for jn, sc_names in junction_to_scs.items():
        lag = junction_lag.get(jn, 0.0)
        for sc in sc_names:
            sc_lags[sc] = lag
        print(f"  {jn:<22s}  lag = {lag * 24:.2f} h  ({len(sc_names)} SC(s))")

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(sc_lags, f, indent=2)
        print(f"  network_lags.json → {save_path}")

    return sc_lags


def bc_node_lag(
    inp_path: str,
    node_id: str,
    outlet_node: str,
    method: str = "constant",
    velocity_fps: float = 2.0,
) -> float:
    """
    Travel-time lag (days) from a single node to *outlet_node*.

    For a boundary-condition (BC) inflow entering the model at *node_id*
    rather than through a subcatchment's outlet — e.g. an upstream
    already-calibrated area's flow, injected for downstream multi-gage
    calibration (see ``CalOptimizer.load_bc_inflow(bc_lag_days=...)``).

    Validates that *node_id* AND *outlet_node* are real nodes (``JUNCTIONS``,
    ``OUTFALLS``, ``STORAGE``, or ``DIVIDERS``) in the INP before tracing a
    path: unlike ``compute_network_lags()``/``_trace_path()``, which silently
    return lag ``0.0`` for an unknown node (same as a mistyped subcatchment
    ``outlet``), this raises so a typo'd node fails loudly instead of
    silently calibrating with no lag.  A 0.0 result between two distinct
    valid nodes additionally prints a warning, because it is indistinguishable
    from a missing/reversed flow path.

    Parameters
    ----------
    inp_path : str
        Path to the SWMM INP file.
    node_id : str
        Node name where the BC inflow enters the model.
    outlet_node : str
        SWMM node name at the calibration gage (same value passed to
        ``compute_network_lags``).
    method, velocity_fps
        Same as ``compute_network_lags()``.

    Returns
    -------
    float
        Lag in days (0.0 if *node_id* is *outlet_node* or has no path to it;
        the no-path case prints a warning).

    Raises
    ------
    ValueError
        If *node_id* or *outlet_node* is not a JUNCTIONS/OUTFALLS/STORAGE/
        DIVIDERS node in the INP.
    FileNotFoundError
        If *inp_path* does not exist (no longer masked as an unknown-node
        error).
    """
    from .swmm_parse import scan_swmm_sections

    # No try/except here: scan_swmm_sections returns empty rows for a
    # missing section and only raises on file errors — those must propagate
    # as file errors, not masquerade as an unknown-node ValueError.  One
    # pass collects all four node sections (per-section reads would pay a
    # full scan of the INP for every ABSENT section).
    node_sections = scan_swmm_sections(
        inp_path, ("JUNCTIONS", "OUTFALLS", "STORAGE", "DIVIDERS")
    )
    node_names: set[str] = {
        row[0] for rows in node_sections.values() for row in rows if row
    }
    for arg_name, node in (("node_id", node_id), ("outlet_node", outlet_node)):
        if node not in node_names:
            raise ValueError(
                f"{arg_name} {node!r} is not a JUNCTIONS/OUTFALLS/STORAGE/"
                f"DIVIDERS node in {inp_path!r}. Check for a typo or wrong "
                f"node name."
            )

    sc_df = pd.DataFrame([{"name": "__bc_node_lag__", "outlet": node_id}])
    lags = compute_network_lags(
        inp_path=inp_path,
        outlet_node=outlet_node,
        sc_df=sc_df,
        method=method,
        velocity_fps=velocity_fps,
        save_path=None,
    )
    lag = lags["__bc_node_lag__"]
    if lag == 0.0 and node_id != outlet_node:
        # Both nodes exist, but a 0.0 lag between distinct nodes means either
        # the path is all zero-length hydraulic links (genuine ~0 travel
        # time) or NO directed path exists (reversed conduit digitization,
        # swapped arguments) — _trace_path returns an empty path silently in
        # that case.  Be loud so a wiring mistake is not read as "no lag".
        print(
            f"WARNING: bc_node_lag({node_id!r} -> {outlet_node!r}) = 0.0 days. "
            f"This is correct only if the BC node drains to the gage through "
            f"zero-length links; it is ALSO what a missing/reversed flow path "
            f"returns. Verify the node connects downstream to {outlet_node!r}."
        )
    return lag


# ---------------------------------------------------------------------------
# Production SWMM run preparation
# ---------------------------------------------------------------------------

def prepare_swmm_run(
    model,
    cal,
    inp_path: str,
    artifacts_dir: str,
    sim_start: str,
    sim_end: str,
    warmup_buffer: int = 365,
    flow_unit: str = "cfs",
) -> str:
    """
    Generate per-SC RDII .tsf files and a patched SWMM INP ready for PCSWMM.

    Parameters
    ----------
    model : RdiiModel
        The calibrated RDII model.
    cal : CalOptimizer
        Optimizer with best_params set (i.e. after optimize()).
    inp_path : str
        Path to the original SWMM INP file.
    artifacts_dir : str
        Output directory.  TSF files go to ``artifacts_dir/tsf/``.
    sim_start, sim_end : str
        Simulation window bounds (``'YYYY-MM-DD'`` format).
    warmup_buffer : int
        Days of warm-up to prepend (excluded from TSF output).
    flow_unit : str
        Flow unit label written in TSF header (default ``'cfs'``).

    Returns
    -------
    str
        Path to the patched INP file ready to open in PCSWMM.
    """
    from .io import write_tsf, write_swmm_inflows, write_swmm_timeseries

    tsf_dir = os.path.join(artifacts_dir, "tsf")
    os.makedirs(tsf_dir, exist_ok=True)

    warmup_start = pd.Timestamp(sim_start) - pd.Timedelta(days=warmup_buffer)

    cal_models = cal.build_calibrated_models()
    sc_df = model._sc_df

    tsf_map: dict[str, str] = {}
    for sc_model, sc_row in zip(cal_models, sc_df.itertuples()):
        df_sim = model._get_sc_df(
            sc_row=sc_df.loc[sc_df["name"] == sc_row.name].iloc[0],
            observed=model._observed,
            start=warmup_start,
            end=pd.Timestamp(sim_end),
        )
        sc_model.prepare(df_sim)
        pred_sim = sc_model.predict()

        tsf_path = os.path.join(tsf_dir, f"{SH_SERIES_PREFIX}{sc_row.name}.tsf")
        tsf_name = write_tsf(
            pred_df=pred_sim,
            sc_name=sc_row.name,
            out_path=tsf_path,
            sim_start=sim_start,
            sim_end=sim_end,
            flow_unit=flow_unit,
        )
        tsf_map[sc_row.name] = tsf_name

    sc_records = sc_df[["name", "outlet"]].to_dict(orient="records")

    inflows_path = os.path.join(artifacts_dir, "swmm_inflows.txt")
    timeseries_path = os.path.join(artifacts_dir, "swmm_timeseries.txt")

    write_swmm_inflows(sc_records=sc_records, tsf_map=tsf_map, out_path=inflows_path)
    write_swmm_timeseries(tsf_map=tsf_map, tsf_dir=tsf_dir, out_path=timeseries_path)

    # Build section content strings from the written snippet files.
    with open(inflows_path, "r", encoding="utf-8") as f:
        inflows_content = f.read()
    with open(timeseries_path, "r", encoding="utf-8") as f:
        timeseries_content = f.read()

    # Strip the section header lines from the snippet files (patch_inp re-adds them).
    def _strip_header(text: str, header: str) -> str:
        lines = text.splitlines(keepends=True)
        return "".join(ln for ln in lines if not ln.strip().startswith(header.strip("[]").split("]")[0].split("[")[-1]))

    # patch_inp expects content WITHOUT the [SECTION] header line.
    inflows_body = "\n".join(
        ln for ln in inflows_content.splitlines() if not ln.startswith("[")
    ) + "\n"
    timeseries_body = "\n".join(
        ln for ln in timeseries_content.splitlines() if not ln.startswith("[")
    ) + "\n"

    # Build OPTIONS patch: update simulation dates.
    sections = _parse_inp_sections(inp_path)
    opts = sections.get("[OPTIONS]", "")
    sim_start_swmm = pd.Timestamp(sim_start).strftime("%m/%d/%Y")
    sim_end_swmm = pd.Timestamp(sim_end).strftime("%m/%d/%Y")
    opts = re.sub(r"(?m)^(START_DATE\s+).*$", rf"\g<1>{sim_start_swmm}", opts)
    opts = re.sub(r"(?m)^(END_DATE\s+).*$", rf"\g<1>{sim_end_swmm}", opts)
    opts = re.sub(r"(?m)^(REPORT_START_DATE\s+).*$", rf"\g<1>{sim_start_swmm}", opts)

    out_name = os.path.splitext(os.path.basename(inp_path))[0]
    patched_inp = os.path.join(artifacts_dir, f"{out_name}_swmmrdii.inp")

    patch_inp(
        inp_path=inp_path,
        out_path=patched_inp,
        replacements={
            "[INFLOWS]": inflows_body,
            "[TIMESERIES]": timeseries_body,
            "[OPTIONS]": opts,
        },
        # Merge [INFLOWS] and [TIMESERIES] so pre-existing entries (DWF,
        # other lateral inflows) are preserved alongside the SH_ rows.
        merge_sections={"[INFLOWS]", "[TIMESERIES]"},
    )

    print(f"  TSF files   → {tsf_dir}/  ({len(tsf_map)} subcatchments)")
    print(f"  [INFLOWS]   → {inflows_path}")
    print(f"  [TIMESERIES]→ {timeseries_path}")
    print(f"  Patched INP → {patched_inp}")
    return patched_inp


# ---------------------------------------------------------------------------
# Coupled sparsehydro + SWMM run (production / PCSWMM integration)
# ---------------------------------------------------------------------------

def _retag_out_version(out_path: str, target_version) -> None:
    """
    Re-stamp an openswmm binary output as an earlier SWMM 5.x version so
    PCSWMM registers it as current results.

    openswmm's engines write version 53000 (legacy solver) or 60000 (v6
    engine) with renumbered input-property type codes (the code renumbering
    is identical for both).  PCSWMM can *read* those files (plots work) but
    does not *register* them — the status bar shows "No results available".
    Apart from the version stamp and property codes the layout is
    byte-identical to a 5.2.4 output (verified by running the same model
    through openswmm and PCSWMM's native engine and comparing the files),
    so rewriting those fields makes the file fully native to PCSWMM.

    *target_version* is the SWMM version the user's PCSWMM is configured to
    run, given as ``"5.2.4"`` or the integer stamp ``52004``.  It is a
    user-declared label: the run itself was computed by openswmm's engine
    (backwards-compatible with SWMM 5.x models), and the stamp tells PCSWMM
    which format family to treat the file as.

    No-op for files that are not openswmm output (version 53000 or 60000).
    """
    import struct

    MAGIC = 516114522
    if isinstance(target_version, str):
        parts = [int(p) for p in target_version.split(".")]
        while len(parts) < 3:
            parts.append(0)
        target_version = parts[0] * 10000 + parts[1] * 1000 + parts[2]

    # 5.3 -> 5.0/5.1/5.2 input-property enum renumbering (enum was stable
    # across 5.0-5.2, so one map covers all retag targets).
    NODE_CODE_MAP = {4: 3}                     # max depth
    LINK_CODE_MAP = {6: 4, 7: 4, 4: 3, 8: 5}   # offset1, offset2, max depth, length

    with open(out_path, "r+b") as f:
        magic, version, _units, n_sub, n_node, n_link, _poll = struct.unpack(
            "<7i", f.read(28)
        )
        if magic != MAGIC or version not in (53000, 60000):
            return
        f.seek(-24, 2)
        _id_pos, prop_pos, _res_pos, _n_per, _err, magic2 = struct.unpack(
            "<6i", f.read(24)
        )
        if magic2 != MAGIC:
            return

        def _remap_codes(n_entities: int, code_map: dict) -> None:
            (k,) = struct.unpack("<i", f.read(4))
            pos = f.tell()
            codes = list(struct.unpack(f"<{k}i", f.read(4 * k)))
            new = [code_map.get(c, c) for c in codes]
            if new != codes:
                f.seek(pos)
                f.write(struct.pack(f"<{k}i", *new))
            f.seek(pos + 4 * k + 4 * k * n_entities)

        f.seek(prop_pos)
        _remap_codes(n_sub, {})              # subcatchment codes unchanged
        _remap_codes(n_node, NODE_CODE_MAP)
        _remap_codes(n_link, LINK_CODE_MAP)

        f.seek(4)
        f.write(struct.pack("<i", int(target_version)))


def _run_swmm_dll(
    dll_path: str,
    inp_path: str,
    rpt_path: str,
    out_path: str,
    progress_callback=None,
    total_days: Optional[float] = None,
) -> int:
    """
    Run a SWMM 5.x engine DLL (EPA swmm5 API) via ctypes.

    Optional alternative to the default openswmm solver: run the hydraulics
    through an actual SWMM 5.x engine build so the .out is native to that
    engine with no retagging (see ``_retag_out_version`` for the default
    approach).  Note the DLL must be 64-bit — the engines bundled with
    PCSWMM are 32-bit and cannot be loaded here; an official EPA x64 build
    is required.  Runtime depends on how the DLL was compiled (optimization
    flags matter more than the 5.x version number).

    Progress is reported through *progress_callback* (fraction 0..1) by
    advancing the simulation one day per ``swmm_stride`` call.  Returns
    the engine error code (0 = success), mirroring openswmm's run_solver.
    """
    import ctypes

    os.add_dll_directory(os.path.dirname(os.path.abspath(dll_path)))
    dll = ctypes.CDLL(dll_path)

    dll.swmm_open.argtypes = [ctypes.c_char_p] * 3
    dll.swmm_open.restype = ctypes.c_int
    dll.swmm_start.argtypes = [ctypes.c_int]
    dll.swmm_start.restype = ctypes.c_int
    dll.swmm_stride.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
    dll.swmm_stride.restype = ctypes.c_int
    for _fn in ("swmm_end", "swmm_report", "swmm_close"):
        getattr(dll, _fn).restype = ctypes.c_int

    err = dll.swmm_open(
        inp_path.encode("mbcs"), rpt_path.encode("mbcs"), out_path.encode("mbcs")
    )
    if not err:
        err = dll.swmm_start(1)
        if not err:
            elapsed = ctypes.c_double(0.0)
            while True:
                err = dll.swmm_stride(86400, ctypes.byref(elapsed))
                if err or elapsed.value <= 0.0:
                    break
                if progress_callback is not None and total_days:
                    progress_callback(min(elapsed.value / total_days, 1.0))
        dll.swmm_end()
    if not err:
        dll.swmm_report()
    dll.swmm_close()
    return err


def _reset_creation_time(path: str) -> None:
    """
    Set a file's Windows creation time to now.

    Replacing a file with ``os.replace`` (or overwriting it in place) keeps
    the *old* file's creation time due to NTFS filename tunneling.  PCSWMM's
    ``refresh_output()`` compares the .out creation time against the model
    and rejects it with "Output file is older" if it predates the INP.
    """
    if os.name != "nt":
        return
    import ctypes
    import time

    kernel32 = ctypes.windll.kernel32
    kernel32.CreateFileW.restype = ctypes.c_void_p
    FILE_WRITE_ATTRIBUTES = 0x0100
    FILE_SHARE_ALL = 0x7  # read | write | delete
    OPEN_EXISTING = 3
    INVALID_HANDLE = ctypes.c_void_p(-1).value

    handle = kernel32.CreateFileW(
        path, FILE_WRITE_ATTRIBUTES, FILE_SHARE_ALL, None, OPEN_EXISTING, 0, None
    )
    if handle in (None, INVALID_HANDLE):
        return
    try:
        # FILETIME: 100-ns ticks since 1601-01-01 UTC.
        ticks = ctypes.c_ulonglong(int((time.time() + 11644473600) * 10_000_000))
        kernel32.SetFileTime(ctypes.c_void_p(handle), ctypes.byref(ticks), None, None)
    finally:
        kernel32.CloseHandle(ctypes.c_void_p(handle))


def _index_pkl_bundles(pkl_dirs) -> dict[str, str]:
    """
    Recursively index ``SH_<sc_name>.pkl`` bundles across one or more
    directories.

    Accepts a single directory path or a list of them; each is walked
    recursively, so bundles organized into per-gage subfolders (e.g.
    ``sc_models/03145483/SH_*.pkl``) are found without flattening.

    Returns
    -------
    dict
        Mapping of bundle filename (``'SH_<sc_name>.pkl'``) → full path.

    Raises
    ------
    FileNotFoundError
        If any of the given directories does not exist.
    ValueError
        If the same bundle filename appears in more than one location —
        the same subcatchment calibrated in two areas is ambiguous, and
        picking one silently could double-represent or mis-represent it.
    """
    if isinstance(pkl_dirs, (str, os.PathLike)):
        pkl_dirs = [pkl_dirs]
    index: dict[str, str] = {}
    for d in pkl_dirs:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"PKL directory not found: {os.path.abspath(d)}")
        for root, _dirs, files in os.walk(d):
            for f in files:
                if f.startswith(SH_SERIES_PREFIX) and f.endswith(".pkl"):
                    path = os.path.join(root, f)
                    prev = index.get(f)
                    if prev is not None and not os.path.samefile(prev, path):
                        raise ValueError(
                            f"Duplicate PKL bundle {f!r} found in two locations:\n"
                            f"  {prev}\n"
                            f"  {path}\n"
                            f"The same subcatchment appears to be calibrated in "
                            f"more than one area. Remove one copy or pass a "
                            f"narrower pkl_dir."
                        )
                    index[f] = path
    return index


def check_inp_window(
    inp_path: str,
    required_start,
    required_end,
    warmup_days: int = 0,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Verify the INP simulation window covers a required scoring window.

    Intended as a pre-flight guard before ``run_sparsehydro()``: the coupled
    run simulates the [START_DATE, END_DATE] window from the INP ``[OPTIONS]``
    block, so if that window does not cover the calibration/validation
    scoring period (CAL_START through VAL_END in the CalVal notebooks), the
    routed output is silently truncated and the section-8b/8c metrics score
    only the overlap.

    Parameters
    ----------
    inp_path : str
        SWMM .inp whose ``[OPTIONS]`` START_DATE/START_TIME and
        END_DATE/END_TIME are checked (missing times default to 00:00:00).
    required_start, required_end : str or datetime-like
        The window the simulation must cover (e.g. ``CAL_START``, ``VAL_END``).
    warmup_days : int
        Warm-up lead-in required ahead of *required_start* (default 0).
        Pass the notebook's ``WARMUP_BUFFER`` so START_DATE must be at or
        before ``required_start - warmup_days``, guaranteeing the simulated
        window itself includes the warm-up period before the first scored day.

    Returns
    -------
    tuple[pd.Timestamp, pd.Timestamp]
        ``(sim_start, sim_end)`` parsed from the INP (dates + times), for
        further use.

    Raises
    ------
    ValueError
        If START_DATE/END_DATE are missing or the simulated window falls
        short of full-day coverage of
        [required_start - warmup_days, required_end].  Note END_TIME counts:
        ``END_DATE 12/31/2025`` + ``END_TIME 00:00:00`` simulates NONE of
        12/31 and fails a ``required_end`` of 2025-12-31.
    """
    from .swmm_parse import scan_swmm_sections

    # One early-exit scan of just [OPTIONS] (it sits at the top of every
    # INP) instead of buffering the whole file via _parse_inp_sections —
    # PCSWMM exports with embedded [TIMESERIES] run to hundreds of MB.
    opt_rows = scan_swmm_sections(inp_path, ("OPTIONS",))["OPTIONS"]
    opts = {row[0].upper(): row[1] for row in opt_rows if len(row) >= 2}

    def _opt(key):
        return opts.get(key)

    def _time_delta(timestr):
        # SWMM times may be "24:00:00", which pd.Timestamp cannot parse —
        # convert HH[:MM[:SS]] to a Timedelta added onto the date instead.
        parts = [int(p) for p in timestr.split(":")] + [0, 0]
        return pd.Timedelta(hours=parts[0], minutes=parts[1], seconds=parts[2])

    start_str = _opt("START_DATE")
    end_str   = _opt("END_DATE")
    if not start_str or not end_str:
        raise ValueError(
            f"Could not find START_DATE / END_DATE in [OPTIONS] of {inp_path!r}"
        )
    # END_TIME matters: a simulation ending at 00:00:00 on END_DATE covers
    # NONE of that day, so a bare-date comparison passes while the last
    # scored day is silently missing from the .out.
    sim_start = pd.Timestamp(start_str) + _time_delta(_opt("START_TIME") or "00:00:00")
    sim_end   = pd.Timestamp(end_str)   + _time_delta(_opt("END_TIME")   or "00:00:00")
    # Daily scoring needs FULL days: coverage from required_start 00:00
    # through the END of the required_end day (= required_end + 1 day, 00:00).
    req_end   = pd.Timestamp(required_end).normalize() + pd.Timedelta(days=1)
    req_start = (
        pd.Timestamp(required_start).normalize()
        - pd.Timedelta(days=int(warmup_days))
    )
    warmup_note = (
        f" (= start {pd.Timestamp(required_start):%m/%d/%Y} - "
        f"{int(warmup_days)}-day warm-up)" if warmup_days else ""
    )
    one_day = pd.Timedelta(days=1)

    problems, fixes = [], []
    if sim_start > req_start:
        missing = -((req_start - sim_start) // one_day)   # ceil of positive gap
        problems.append(
            f"simulation starts {sim_start:%m/%d/%Y %H:%M}, AFTER the required "
            f"start ({req_start:%m/%d/%Y}{warmup_note}) - the first "
            f"{missing} required day(s) would be missing or incomplete"
        )
        fixes.append(
            f"set START_DATE to {req_start:%m/%d/%Y} (or earlier) with "
            f"START_TIME 00:00:00"
        )
    if sim_end < req_end:
        missing = -((sim_end - req_end) // one_day)       # ceil of positive gap
        problems.append(
            f"simulation ends {sim_end:%m/%d/%Y %H:%M}, BEFORE the end of the "
            f"required last day ({pd.Timestamp(required_end):%m/%d/%Y}, i.e. "
            f"coverage through {req_end:%m/%d/%Y %H:%M}) - the last "
            f"{missing} scored day(s) would be missing or incomplete"
        )
        fixes.append(
            f"set END_DATE to {req_end:%m/%d/%Y} with END_TIME 00:00:00"
        )
    if problems:
        # Name the field that is actually wrong.  Suggesting END_DATE when the
        # START is short sends the reader to the wrong setting.
        raise ValueError(
            "INP simulation window does not cover the scoring window"
            + (" + warm-up" if warmup_days else "") + ".\n"
            f"  INP      : {sim_start:%m/%d/%Y %H:%M} -> {sim_end:%m/%d/%Y %H:%M}  ({inp_path})\n"
            f"  Required : {req_start:%m/%d/%Y}{warmup_note} -> end of {pd.Timestamp(required_end):%m/%d/%Y}\n"
            "  - " + "\n  - ".join(problems) + "\n"
            "Fix in the SWMM [OPTIONS] (PCSWMM: Simulation Options > Dates):\n"
            "  - " + "\n  - ".join(fixes) + "\n"
            "REPORT_START_DATE is rewritten automatically by the sparsehydro "
            "run, so it needs no manual change.\n"
            "Note the warm-up only needs RAINFALL/TEMPERATURE coverage, not "
            "observed flow — extending START_DATE earlier costs nothing as "
            "long as the rain and temperature series reach back that far."
        )

    print(
        f"INP window OK : {sim_start:%m/%d/%Y %H:%M} -> {sim_end:%m/%d/%Y %H:%M} "
        f"covers required {req_start:%m/%d/%Y}{warmup_note} -> "
        f"end of {pd.Timestamp(required_end):%m/%d/%Y}"
    )
    return sim_start, sim_end


def write_series_tsf(
    path: str,
    series: dict,
    start=None,
    end=None,
    unit: str = "cfs",
    col_label: str = "Flow",
) -> int:
    """
    Write named time series to a PCSWMM ``.tsf`` file, one column each.

    Produces the tab-delimited single-block layout PCSWMM exports and
    :func:`swmmrdii.forcing.load_tsf` reads::

        IDs:<TAB>NameA<TAB>NameB
        Date/Time<TAB>Flow<TAB>Flow
        M/d/yyyy<TAB>cfs<TAB>cfs
        1/2/2000 12:00:00 AM<TAB>0.0000<TAB>1.2500

    All series are aligned onto the union of their timestamps; gaps become
    0.0 so every row is rectangular, as PCSWMM requires.

    Parameters
    ----------
    path : str
        Output ``.tsf`` path.  Parent directories are created.
    series : dict of {str: pd.Series}
        Series keyed by the identifier to write in the ``IDs:`` header.
        Each must be indexed by datetime.  Columns are written in sorted
        key order so the file is reproducible.
    start, end : datetime-like or None
        Optional inclusive clip applied to the time index.
    unit : str
        Unit label for the third header line (default ``'cfs'``).
    col_label : str
        Column-type label for the second header line (default ``'Flow'``).

    Returns
    -------
    int
        Number of series (columns) written.
    """
    if not series:
        raise ValueError("write_series_tsf: `series` is empty — nothing to write.")

    names = sorted(series)
    mat = pd.DataFrame({n: series[n] for n in names}).sort_index()
    if start is not None:
        mat = mat[mat.index >= start]
    if end is not None:
        mat = mat[mat.index <= end]
    mat = mat.fillna(0.0)

    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)

    times = pd.DatetimeIndex(mat.index)
    vals  = mat.to_numpy()

    with open(path, "w", encoding="utf-8") as f:
        f.write("IDs:\t" + "\t".join(names) + "\n")
        f.write("Date/Time\t" + "\t".join(col_label for _ in names) + "\n")
        f.write("M/d/yyyy\t" + "\t".join(unit for _ in names) + "\n")
        buf: list[str] = []
        for i, t in enumerate(times):
            # 12-hour clock with AM/PM and unpadded month/day — matches the
            # "M/d/yyyy" format declared on the third header line.
            h12  = t.hour % 12 or 12
            ampm = "AM" if t.hour < 12 else "PM"
            stamp = (f"{t.month}/{t.day}/{t.year} "
                     f"{h12:02d}:{t.minute:02d}:{t.second:02d} {ampm}")
            buf.append(stamp + "\t" + "\t".join(f"{v:.4f}" for v in vals[i]))
            if len(buf) >= 50_000:
                f.write("\n".join(buf) + "\n")
                buf.clear()
        if buf:
            f.write("\n".join(buf) + "\n")

    return len(names)


def run_sparsehydro(
    inp_path: str,
    pkl_dir: Optional[str | list[str]] = None,
    out_inp: Optional[str] = None,
    bfi_field: str = "BFI",
    imperv_field: str = "Imperv",
    temp_field: str = "TempGage",
    temp_unit: str = "degF",
    network_lags: Optional[dict] = None,
    out_dir: Optional[str] = None,
    flow_unit: str = "cfs",
    show_progress: bool = True,
    out_file: Optional[str] = None,
    n_jobs: Optional[int] = None,
    engine: str = "legacy",
    engine_dll: Optional[str] = None,
    out_version: Optional[str] = None,
    in_place: bool = False,
    auto_rescale: bool = True,
    write_sc_tsf: bool = True,
    sc_tsf_path: Optional[str] = None,
) -> tuple[str, str]:
    """
    Full coupled sparsehydro + SWMM run from pkl bundles and a SWMM INP.

    Reads the simulation window directly from the INP ``[OPTIONS]`` block
    (START_DATE / END_DATE), so no dates need to be passed separately.
    Precipitation and temperature forcing are resolved from the INP exactly
    as ``RdiiModel.from_swmm()`` does — any changes to rain gage or
    temperature time-series references in the INP are picked up automatically.

    Steps
    -----
    1. Parse INP for simulation dates, subcatchment→junction mapping, and
       forcing references (rain gages, temperature time-series).
    2. Load one cloudpickle bundle per subcatchment from *pkl_dir*
       (``SH_<sc_name>.pkl``).
    3. Predict per-SC RDII over the full sim window using the calibrated
       sparsehydro model with the current INP forcing.
    4. Sum predictions per outlet junction; apply network lags if provided.
    5. Write a single combined SWMM5 interface file (all junctions), plus a
       PCSWMM TSF holding each subcatchment's RDII series separately.
    6. Write a routing-only patched INP referencing the interface file.
    7. Run SWMM via ``run_solver()`` — no Python time-stepping.
    8. Return paths to the patched INP and the binary ``.out`` file.

    Parameters
    ----------
    inp_path : str
        Path to the SWMM ``.inp`` model file.  Simulation dates, rain gage
        references, and temperature time-series are all read from here.
    pkl_dir : str, list of str, or None
        Directory (or list of directories) containing ``SH_<sc_name>.pkl``
        cloudpickle bundles produced by ``CalOptimizer.save_sc_models()``.
        Each directory is searched **recursively**, so bundles kept in
        per-gage subfolders (e.g. ``sc_models/<gage>/SH_*.pkl``) are found
        without flattening; a list combines several calibrated areas
        (upstream + local) into one run.  The same bundle filename appearing
        in more than one location raises ``ValueError``.  Defaults to
        ``<inp_stem>.pkl/`` in the same directory as *inp_path*
        (e.g. ``MyModel.inp`` → ``MyModel.pkl/``).
    out_inp : str or None
        Path for the patched routing-only INP written by this function.
        Defaults to ``<inp_stem>_sparsehydro.inp`` next to the original.
        Must not be *inp_path* itself (``ValueError``) — the original
        model is never modified.
    bfi_field : str
        PCSWMM DB user-field to read LIVE per-SC BFI values from (default
        ``'BFI'``).  This is the value source for run-time auto-rescaling
        and need not be the field used at calibration — point it at a
        scenario attribute (e.g. a future-condition BFI column) and the
        difference from the calibration-time value is applied via
        ``rescale_bfi()``.  A note is printed when it differs from the
        bundle's ``bfi_field_used``.
    imperv_field : str
        Subcatchment attribute to read LIVE ``%Imperv`` values from
        (default ``'Imperv'``).  Resolved against the INP's native
        ``[SUBCATCHMENTS]`` columns first (``'Imperv'``/``'imperv'`` is
        the INP's own %Imperv column; ``width`` and ``slope`` are also
        available), then the PCSWMM DB user-fields.  Like *bfi_field*,
        this is a value source, not a consistency check — point it at a
        future-scenario attribute to run land-use scenarios without
        re-calibrating (the difference is applied via
        ``rescale_imperv()``).  A note is printed when it differs from
        the bundle's ``imperv_field_used``.
    temp_field : str
        DB attribute naming the temperature ``[TIMESERIES]`` per SC
        (default ``'TempGage'``).
    temp_unit : str
        Unit of the temperature data (default ``'degF'``).
    auto_rescale : bool
        When True (default), each subcatchment's bundle parameters are
        adjusted at run time to the CURRENT SWMM attributes before
        predicting (identity when nothing changed):

        * ``area_acres`` — always taken from the live INP [SUBCATCHMENTS].
        * BFI — when the PCSWMM ``.db`` is present, R_3 is rescaled via
          ``rescale_bfi()`` if the live value differs from the
          calibration-time value stored in the bundle.
        * %imperviousness — when the ``.db`` is present and the
          *imperv_field* column exists, ia_max is rescaled via
          ``rescale_imperv()`` if the live value differs.

        This is what makes donor bundles work: copy a calibrated
        ``SH_<donor>.pkl`` to ``SH_<target>.pkl`` and the target SC's own
        area/BFI/imperviousness are applied automatically.  Bundles written
        before BFI-rescale support must be backfilled with
        ``scripts/patch_pkl_bfi.py`` before a *differing* BFI can be
        applied (a clear error is raised otherwise; unchanged values never
        require the backfill).

        If a bundle records that it was calibrated WITH a BFI or
        imperviousness field but the live value cannot be read from the
        current model (no ``.db`` next to the INP, or the attribute was
        removed), a ``ValueError`` is raised rather than silently running
        with stale calibration-time values.  Legacy bundles (no
        ``params_v2``) are used verbatim with a printed note.
        Set False to use every bundle exactly as saved.
    network_lags : dict or None
        Deprecated — no longer used. Network lags are applied during calibration
        so sparseHydro-only metrics are evaluated at the gage. When running with
        SWMM, inflows are injected at each SC's outlet junction and hydraulic
        routing provides the travel time naturally; applying the lag a second time
        would double-count it. Parameter retained for API compatibility.
    out_dir : str or None
        Directory for the combined TSF and output files.  Defaults to a
        ``sparsehydro/`` subfolder next to the INP.
    flow_unit : str
        Flow unit label in the TSF header (default ``'cfs'``).
    write_sc_tsf : bool
        When True (default), also write a PCSWMM ``.tsf`` holding one
        column per subcatchment.  The SWMM interface file carries only
        junction totals, so a junction fed by several subcatchments hides
        the individual contributions; this file exposes each one for
        inspection in PCSWMM.  Set False to skip it.
    sc_tsf_path : str or None
        Explicit path for that per-subcatchment TSF.  Defaults to
        ``<inp_stem>_sparsehydro.tsf`` inside *out_dir*.
    warmup_days : int
        Number of days to prepend before the INP START_DATE for the SWMM
        hydraulic run (default 365).  The inflows interface file covers
        [START_DATE - warmup_days, END_DATE] so pipe storage and junction
        depths are initialised before the scored window begins.  Set to 0
        to match the INP window exactly.
    show_progress : bool
        If ``True`` (default), print a live ``0%–100%`` progress line while
        SWMM is running via ``run_solver``'s progress callback.
    out_file : str or None
        Explicit path for the SWMM binary output.  Defaults to
        ``<patched_inp_stem>.out`` next to the patched INP.  Must not be
        *inp_path* itself (``ValueError``).
    n_jobs : int or None
        Number of threads for per-subcatchment RDII prediction.  Defaults
        to ``cpu_count - 1``.  Pass ``1`` to force sequential prediction.
    engine : str
        Which openswmm engine runs the hydraulics:

        * ``"legacy"`` (default) — the EPA SWMM 5.x solver preserved in the
          openswmm package (its core identifies as SWMM 5.3 beta).
        * ``"v6"`` — the refactored OpenSWMM 6 engine
          (``openswmm.engine``).  Beta — currently FAILS validation for
          this workflow.  6.0.0.dev2 silently ignored both the
          ``USE INFLOWS`` interface file and ``IGNORE_RAINFALL``;
          6.0.0a2 (validated 2026-07-08) FIXED the interface file but
          still ignores ``IGNORE_RAINFALL``, so runs on a patched INP
          contain native runoff on top of the sparsehydro inflows.
          Kept as an option for re-validation against future releases.

        Both write the classic binary .out format with the full entity set
        and both need *out_version* for PCSWMM to register the results.
        Ignored when *engine_dll* is set.
    engine_dll : str or None
        Path to a SWMM 5.x engine DLL (EPA swmm5 API, x64).  When set, the
        hydraulic run uses that engine instead of openswmm and the .out is
        native to that engine (no retag applied).  When ``None`` (default),
        the run uses the openswmm engine selected by *engine*; pair with
        *out_version* for PCSWMM compatibility.
    out_version : str or None
        SWMM version to stamp into the openswmm .out (e.g. ``"5.2.4"``),
        matching the version the user's PCSWMM is configured to run.
        openswmm's 5.3-beta stamp (53000) is not registered by PCSWMM
        ("No results available" in the status bar) even though the file
        layout is otherwise identical to 5.x output.  This is a
        user-declared label — the run is still computed by openswmm's
        engine, which is backwards-compatible with SWMM 5.x models.
        Ignored when *engine_dll* is set; ``None`` leaves the stamp as-is.
        See ``_retag_out_version``.
    in_place : bool
        If ``True``, adopt the results under the ORIGINAL model's stem
        after the run: ``<inp_stem>.out`` / ``<inp_stem>.rpt`` are
        overwritten next to *inp_path* (creation time reset so PCSWMM
        registers them — any previous run's results at those paths are
        replaced), and the run scaffolding (the patched routing-only INP
        and the inflows interface file) is deleted.  Opening the original
        model in PCSWMM then shows the sparsehydro results directly, with
        no ``*_sparsehydro`` files left behind.  Safe by construction: the
        patched INP keeps the full entity set, so the ``.out`` matches the
        original model.  Mutually exclusive with *out_file*.  Default
        ``False`` (keep the patched INP + interface file; results stay at
        the patched stem).

    Returns
    -------
    tuple[str, str]
        ``(patched_inp_path, out_file_path)`` — the routing-only INP and
        the SWMM binary output file, both suitable for opening in PCSWMM.
        With ``in_place=True`` the first element is *inp_path* itself (the
        patched INP no longer exists) and the second is ``<inp_stem>.out``.
    """
    from .model import RdiiModel

    try:
        import cloudpickle
    except ImportError:
        raise ImportError("cloudpickle is required: pip install cloudpickle")

    if engine not in ("legacy", "v6"):
        raise ValueError(f"engine must be 'legacy' or 'v6', got {engine!r}")
    if in_place and out_file is not None:
        raise ValueError(
            "in_place=True adopts results as <inp_stem>.out next to the "
            "original INP — it cannot be combined with an explicit out_file."
        )
    # The original model is never modified: all patching happens on copies.
    # Refuse destination paths that would overwrite (and, with in_place=True,
    # afterwards delete) the source INP.
    def _same_path(a, b):
        # normcase: Windows paths are case-insensitive.
        return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))

    for arg_name, arg_path in (("out_inp", out_inp), ("out_file", out_file)):
        if arg_path is not None and _same_path(arg_path, inp_path):
            raise ValueError(
                f"{arg_name} must not be the source INP ({inp_path!r}) — "
                "the original model is never modified. Leave it as None or "
                "point it at a different path."
            )
    if engine == "v6" and engine_dll is None:
        # Validation findings on the v6 engine (pulse test: constant BC inflow
        # via interface file + rainy subcatchment under IGNORE_RAINFALL YES):
        # - 6.0.0.dev2 (Jul 2026): silently (a) never applied the [FILES]
        #   USE INFLOWS interface file (system external inflow = 0 all run)
        #   and (b) ran full rainfall/runoff hydrology despite
        #   IGNORE_RAINFALL YES.
        # - 6.0.0a2 (2026-07-08): (a) is FIXED — interface inflow arrives
        #   exactly; (b) still broken — runoff pulse rides on top of the
        #   inflows.  Re-validate before trusting a newer release.
        print("WARNING: openswmm v6 (validated on 6.0.0a2) does not honor")
        print("         IGNORE_RAINFALL - the patched INP keeps subcatchments,")
        print("         so native runoff will be added ON TOP of sparsehydro")
        print("         inflows. Use engine='legacy' for production runs.")

    run_solver = None
    if engine_dll is None:
        try:
            if engine == "v6":
                from openswmm.engine import run_with_callback as _run_v6
            else:
                from openswmm.solver import run_solver
        except ImportError:
            raise ImportError(
                "openswmm is required: pip install openswmm "
                "(or pass engine_dll= to use a SWMM 5.x engine DLL)"
            )

    # ------------------------------------------------------------------
    # 1. Parse simulation window from INP [OPTIONS].
    # ------------------------------------------------------------------
    sections  = _parse_inp_sections(inp_path)
    opts_text = sections.get("[OPTIONS]", "")

    def _opt(key):
        m = re.search(rf"(?m)^{key}\s+(\S+)", opts_text)
        return m.group(1) if m else None

    start_str = _opt("START_DATE")
    end_str   = _opt("END_DATE")
    if not start_str or not end_str:
        raise ValueError(
            f"Could not find START_DATE / END_DATE in [OPTIONS] of {inp_path!r}"
        )
    sim_start = pd.Timestamp(start_str)
    sim_end   = pd.Timestamp(end_str)

    # Parse WET_STEP (wet-weather runoff timestep) to use as the inflows
    # interface file resolution.  Format is HH:MM:SS; fall back to daily if
    # absent.  Using the INP's own runoff timestep keeps the inflow resolution
    # consistent with the model setup without requiring any user configuration.
    wet_step_str = _opt("WET_STEP")
    if wet_step_str:
        _h, _m, _s = (int(x) for x in wet_step_str.split(":"))
        inflow_dt = pd.Timedelta(hours=_h, minutes=_m, seconds=_s)
    else:
        inflow_dt = pd.Timedelta(days=1)
    inflow_dt_label = f"{inflow_dt.total_seconds() / 60:g} min"

    print(f"Simulation  : {sim_start:%m/%d/%Y} -> {sim_end:%m/%d/%Y}  "
          f"(inflow dt = {inflow_dt_label})")

    # ------------------------------------------------------------------
    # 2. Build RdiiModel from INP + DB to resolve forcing references.
    #    The DB lives alongside the INP with the same stem.
    # ------------------------------------------------------------------
    db_path = os.path.splitext(inp_path)[0] + ".db"
    if not os.path.exists(db_path):
        db_path = None   # no DB — BFI/imperv must come from pkl bundles

    m_obj = RdiiModel.from_swmm(
        inp_path    = inp_path,
        db_path     = db_path,
        bfi_field   = bfi_field,
        temp_field  = temp_field,
        temp_unit   = temp_unit,
    )
    m_obj.load_forcing_from_sc_df()

    # Resolve pkl_dir: default to <inp_stem>.pkl/ next to the INP.  Accepts a
    # single directory (searched recursively — per-gage subfolders are fine)
    # or a list of directories whose bundles are combined for this run.
    if pkl_dir is None:
        inp_stem = os.path.splitext(os.path.basename(inp_path))[0]
        pkl_dir = os.path.join(os.path.dirname(inp_path), f"{inp_stem}.pkl")
    pkl_index = _index_pkl_bundles(pkl_dir)
    _pkl_dirs = [pkl_dir] if isinstance(pkl_dir, (str, os.PathLike)) else list(pkl_dir)
    print(f"PKL folder  : {', '.join(str(d) for d in _pkl_dirs)}"
          f"  [{len(pkl_index)} bundles]")

    # ------------------------------------------------------------------
    # Auto-rescale setup: resolve which LIVE SC attributes are available.
    # Area and the native [SUBCATCHMENTS] attributes (imperv, width,
    # slope) always come from the INP; DB user-fields add to them.  BFI
    # carries real data only when the PCSWMM .db was found — without it,
    # sc_df holds the 1.0 default and rescaling against it would silently
    # corrupt the calibrated parameters.
    # ------------------------------------------------------------------
    _INP_NATIVE_COLS = {"imperv", "width", "slope"}
    bfi_live_ok = auto_rescale and db_path is not None and bfi_field is not None
    imperv_col: Optional[str] = None
    if auto_rescale and imperv_field is not None:
        from .swmm_parse import _sanitize_field_name
        for _cand in (imperv_field, _sanitize_field_name(imperv_field)):
            if _cand in m_obj._sc_df.columns:
                imperv_col = _cand
                break
        if imperv_col is None:
            # Fail-loud policy: a field the caller SPECIFIED must resolve —
            # never silently fall back to calibration-time values.
            raise ValueError(
                f"imperv_field={imperv_field!r} not found among the model's "
                f"subcatchment attributes (checked {imperv_field!r} and "
                f"{_sanitize_field_name(imperv_field)!r} against the INP "
                f"[SUBCATCHMENTS] columns and DB user-fields).\n"
                f"Available columns: {sorted(m_obj._sc_df.columns)}\n"
                f"Pass a valid field name ('Imperv' selects the INP's native "
                f"%Imperv column), or imperv_field=None to run without a "
                f"live %imperv source."
            )
    if auto_rescale:
        if bfi_live_ok:
            bfi_src = f"live DB[{bfi_field!r}]"
        elif db_path is None:
            bfi_src = "calibration-time values (no .db next to INP)"
        else:
            bfi_src = "calibration-time values (bfi_field=None)"
        if imperv_col:
            _src = "INP" if imperv_col in _INP_NATIVE_COLS else "DB"
            imp_src = f"live {_src}[{imperv_col!r}]"
        elif imperv_field is None:
            imp_src = "calibration-time values (imperv_field=None)"
        else:
            imp_src = (f"calibration-time values (field {imperv_field!r} "
                       f"not found in INP [SUBCATCHMENTS] or DB)")
        print(f"Auto-rescale: area=live INP | bfi={bfi_src} | imperv={imp_src}")

    # ------------------------------------------------------------------
    # 3. Load pkl bundles and predict RDII per SC.
    #    Subcatchments without a pkl file are silently skipped — this is
    #    expected during regional calibration where only a subset of SCs
    #    have been calibrated so far.  A summary is printed at the end.
    #    The bundle's stored network_lag_days is NOT applied here (SWMM
    #    routes the injected inflows to the gage); the network_lags
    #    parameter is deprecated and ignored.
    # ------------------------------------------------------------------
    # Field-source notes: the bfi_field/imperv_field arguments are LIVE
    # value sources and may legitimately differ from the fields used at
    # calibration (e.g. a future-scenario attribute).  Auto-rescale applies
    # the live values either way, so a differing name is informational,
    # not an error — printed once per unique combination, not per SC.
    _field_notes_seen: set = set()

    def _note_field_source(kind: str, used, calibrated) -> None:
        key = (kind, used, calibrated)
        if key not in _field_notes_seen:
            _field_notes_seen.add(key)
            if used is None:
                print(f"  NOTE: {kind.lower()}_field=None - live {kind} "
                      f"disabled; bundles were calibrated with {calibrated!r} "
                      f"and run with calibration-time values.")
                return
            action = ("auto-rescale applies the live values"
                      if auto_rescale else
                      "IGNORED - auto_rescale=False uses calibration-time "
                      "values verbatim")
            print(f"  NOTE: {kind} read from field {used!r}; bundles were "
                  f"calibrated with {calibrated!r} - {action}.")

    def _predict_sc(sc_row) -> "tuple[str, Optional[pd.Series], list[str], Optional[dict]]":
        """Load one SC's bundle, auto-rescale it to live SWMM attributes,
        and predict its RDII.  Returns (name, series, rescale_notes, stats)
        or (name, None, [], None) when no PKL bundle exists.  Thread-safe:
        shared state (m_obj caches, _sc_df) is only read; the bundle is
        local."""
        sc_name = sc_row.name
        pkl_path = pkl_index.get(f"{SH_SERIES_PREFIX}{sc_name}.pkl")
        if pkl_path is None:
            return sc_name, None, [], None

        with open(pkl_path, "rb") as f:
            bundle = cloudpickle.load(f)

        # --- Field-source notes (informational, once per combination) ------
        _MISSING = object()
        bundle_bfi    = bundle.get("bfi_field_used",    _MISSING)
        bundle_imperv = bundle.get("imperv_field_used", _MISSING)
        if bundle_bfi is not _MISSING and bfi_field != bundle_bfi:
            _note_field_source("BFI", bfi_field, bundle_bfi)
        if bundle_imperv is not _MISSING and imperv_field != bundle_imperv:
            _note_field_source("imperv", imperv_field, bundle_imperv)
        # -------------------------------------------------------------------

        pv2 = bundle.get("params_v2")
        live_sc = m_obj._sc_df.loc[m_obj._sc_df["name"] == sc_name].iloc[0]

        # --- Auto-rescale to live SWMM attributes --------------------------
        # Compare this SC's current INP/DB attributes against the values
        # frozen into the bundle at calibration time; where they differ,
        # reverse the calibration-time scaling and apply the live value.
        # Unchanged values are left untouched (the rescale is an identity),
        # so bundles for unmodified SCs behave exactly as saved.
        notes: list[str] = []
        if auto_rescale:
            if pv2 is None:
                notes.append("legacy bundle (no params_v2) — auto-rescale skipped")
            else:
                # The bundle records which attributes shaped its calibrated
                # parameters.  If one of them cannot be read from the current
                # model (no .db, or the field was removed), running anyway
                # would silently use stale calibration-time values — the
                # exact failure mode auto-rescale exists to prevent.
                if bundle.get("bfi_field_used") is not None and not bfi_live_ok:
                    # Only reachable with bfi_field=None (a specified field
                    # with no .db already errored while loading the model).
                    raise ValueError(
                        f"Live BFI unavailable for subcatchment '{sc_name}':\n"
                        f"  The bundle was calibrated with BFI from DB field "
                        f"{bundle['bfi_field_used']!r}, but bfi_field=None "
                        f"disables the live BFI source.\n"
                        f"Pass bfi_field naming the DB attribute to read live "
                        f"BFI from (it need not match calibration), or pass "
                        f"auto_rescale=False to run with calibration-time "
                        f"values."
                    )
                if bundle.get("imp_multiplier") and imperv_col is None:
                    # Only reachable with imperv_field=None (a specified but
                    # unresolvable field already errored during setup).
                    raise ValueError(
                        f"Live imperviousness unavailable for subcatchment "
                        f"'{sc_name}':\n"
                        f"  The bundle was calibrated with imperviousness "
                        f"scaling, but imperv_field=None disables the live "
                        f"%imperv source.\n"
                        f"Pass imperv_field naming the attribute to read live "
                        f"%imperv from — 'Imperv' selects the INP's native "
                        f"%Imperv column; a DB user-field name selects that "
                        f"field — or pass auto_rescale=False to run with "
                        f"calibration-time values."
                    )

                live_area = float(live_sc["area_acres"])
                cal_area = float(bundle["area_acres"])
                if not math.isclose(live_area, cal_area, rel_tol=1e-9, abs_tol=1e-12):
                    bundle["area_acres"] = live_area
                    notes.append(f"area {cal_area:g} -> {live_area:g} ac")

                if bfi_live_ok:
                    # Inverse gate: the user armed a live BFI source, but this
                    # bundle was calibrated WITHOUT BFI — its stored bfi/mean_bfi
                    # are 1.0 defaults, so "rescaling" would multiply R_3 by the
                    # raw live BFI value.  Refuse loudly instead.
                    if bundle.get("bfi_field_used") is None:
                        raise ValueError(
                            f"BFI cannot be applied to subcatchment "
                            f"'{sc_name}':\n"
                            f"  bfi_field={bfi_field!r} supplies live BFI "
                            f"values, but the bundle was calibrated WITHOUT "
                            f"BFI (bfi_field_used is None), so there is no "
                            f"BFI scaling to rescale.\n"
                            f"Pass bfi_field=None to run without BFI, or "
                            f"re-calibrate with a BFI field to enable it."
                        )
                    live_bfi = float(live_sc["bfi"])
                    cal_bfi = bundle.get("bfi")
                    bfi_known = isinstance(cal_bfi, (int, float)) and math.isfinite(cal_bfi)
                    if not (bfi_known and math.isclose(live_bfi, cal_bfi,
                                                       rel_tol=1e-9, abs_tol=1e-12)):
                        # Raises with a pointer to scripts/patch_pkl_bfi.py when
                        # the bundle lacks r3_multiplier/mean_bfi metadata.
                        rescale_bfi(bundle, live_bfi)
                        cal_lbl = f"{cal_bfi:g}" if bfi_known else "?"
                        notes.append(f"bfi {cal_lbl} -> {live_bfi:g} (R_3 rescaled)")

                if imperv_col is not None:
                    # Inverse gate: the user armed a live imperviousness source
                    # (the field resolved in the DB), but this bundle carries no
                    # active imperv scaling (imp_multiplier is None/0) — the
                    # calibration never engaged %imperv, so the live values
                    # CANNOT be applied.  Refuse loudly rather than silently
                    # running with imperv-inert parameters.
                    if not bundle.get("imp_multiplier"):
                        raise ValueError(
                            f"%Imperviousness cannot be applied to "
                            f"subcatchment '{sc_name}':\n"
                            f"  imperv_field={imperv_field!r} supplies live "
                            f"%imperv values, but the bundle was calibrated "
                            f"WITHOUT active imperviousness scaling "
                            f"(imp_multiplier is not set) — %imperv never "
                            f"influenced its parameters, so there is no "
                            f"scaling to rescale.\n"
                            f"Re-calibrate with imperv_field to make the "
                            f"parameters imperviousness-responsive, or pass "
                            f"imperv_field=None to knowingly run without "
                            f"%imperv."
                        )
                    raw_imp = live_sc.get(imperv_col)
                    live_imp = float(raw_imp) if pd.notna(raw_imp) else None
                    cal_imp = bundle.get("imperv_pct")
                    imp_known = isinstance(cal_imp, (int, float)) and math.isfinite(cal_imp)
                    if live_imp is not None and not (
                        imp_known and math.isclose(live_imp, cal_imp,
                                                   rel_tol=1e-9, abs_tol=1e-12)
                    ):
                        rescale_imperv(bundle, live_imp)
                        cal_lbl = f"{cal_imp:g}" if imp_known else "?"
                        notes.append(f"imperv {cal_lbl} -> {live_imp:g}% (ia_max rescaled)")
        # -------------------------------------------------------------------

        # Reconstruct the model from the version-independent params_v2 recipe
        # (plain Python dicts — no bytecode, portable across Python versions).
        # Fall back to the pickled model object for legacy bundles that pre-date
        # params_v2 (those were saved on the same Python version, so unpickling works).
        if pv2 is not None:
            from sparsehydro.rdii import make_rtk_ensemble
            sc_model = make_rtk_ensemble(
                area_acres  = float(bundle["area_acres"]),
                n_models    = pv2["n_models"],
                n_triangles = pv2["n_triangles"],
                units       = pv2["units"],
                ia_defaults = pv2["ia_defaults"] or None,
            )
            sc_model.validate()
            sc_model.apply_parameter_values(pv2["params"])
        else:
            sc_model = bundle["model"]

        df_sc = m_obj._get_sc_df(
            sc_row   = live_sc,
            observed = None,
            start    = sim_start,
            end      = sim_end,
        )
        sc_model.prepare(df_sc)
        pred = sc_model.predict().set_index("datetime")["rdii_cfs"]

        # Continuity stats for the run summary: rainfall depth over the sim
        # window, RDII volume, and the post-rescale sum of R fractions.
        sum_r = (sum(v for k, v in pv2["params"].items() if "_R_" in k)
                 if pv2 is not None else float("nan"))
        rain_in = float(df_sc["rainfall_in"].sum()) if "rainfall_in" in df_sc else float("nan")
        if len(pred.index) > 1:
            dt_s = float(pd.Series(pred.index).diff().dropna()
                         .dt.total_seconds().median())
        else:
            dt_s = 86400.0
        stats = {
            "area_acres": float(bundle["area_acres"]),
            "rain_in": rain_in,
            "rdii_ft3": float(pred.clip(lower=0.0).sum()) * dt_s,
            "sum_r": sum_r,
        }

        # No lag shift here — inflows are injected at each SC's outlet junction
        # and SWMM hydraulic routing provides the travel time to the gage naturally.
        # (Lag shift is applied during calibration so sparseHydro-only metrics are
        # evaluated at the gage; it must not be reapplied when SWMM does the routing.)
        return sc_name, pred, notes, stats

    sc_rows = list(m_obj._sc_df.itertuples())
    if n_jobs is None:
        n_jobs = max(1, min(len(sc_rows), (os.cpu_count() or 2) - 1))

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        results = pool.map(_predict_sc, sc_rows)
        if show_progress:
            try:
                # Plain-text tqdm (not tqdm.auto): widget-based bars depend on
                # the frontend's ipywidgets renderer, which VS Code notebooks
                # intermittently fail to connect ("Cannot read properties of
                # undefined (reading 'ipywidgetsKernel')").  Text bars render
                # everywhere.
                from tqdm import tqdm as _tqdm_sc
                results = _tqdm_sc(results, total=len(sc_rows),
                                   desc="Predicting RDII", unit="SC")
            except ImportError:
                pass
        collected = list(results)

    sc_rdii: dict[str, pd.Series] = {n: p for n, p, _, _ in collected if p is not None}
    skipped: list[str] = [n for n, p, _, _ in collected if p is None]

    total_sc = len(m_obj._sc_df)
    print(f"RDII models : {len(sc_rdii)}/{total_sc} subcatchments"
          + (f"  ({len(skipped)} without a PKL bundle - skipped)" if skipped else ""))

    rescaled = [(n, nts) for n, _, nts, _ in collected if nts]
    if rescaled:
        print(f"Auto-rescale: {len(rescaled)} subcatchment(s) adjusted to "
              f"live SWMM attributes:")
        for n, nts in rescaled:
            print(f"    {n}: " + "; ".join(nts))

    # ------------------------------------------------------------------
    # Runoff continuity summary — analogous to SWMM's Runoff Quantity
    # Continuity table.  Checks that predicted RDII volume stays within
    # the mass available from rainfall: sum(R) must be <= 1 per SC, and
    # each SC's capture fraction (RDII vol / rainfall vol) must not exceed
    # its sum(R).  Violations mean mass is being generated.
    # ------------------------------------------------------------------
    stats_list = [(n, st) for n, p, _, st in collected
                  if p is not None and st is not None
                  and math.isfinite(st.get("rain_in", float("nan")))]
    if stats_list:
        SQFT_PER_ACRE, IN_PER_FT, GAL_PER_FT3 = 43560.0, 12.0, 7.48052
        tot_area_ft2 = sum(st["area_acres"] for _, st in stats_list) * SQFT_PER_ACRE
        tot_rain_ft3 = sum(st["rain_in"] / IN_PER_FT * st["area_acres"] * SQFT_PER_ACRE
                           for _, st in stats_list)
        tot_rdii_ft3 = sum(st["rdii_ft3"] for _, st in stats_list)
        rain_in_avg = tot_rain_ft3 / tot_area_ft2 * IN_PER_FT if tot_area_ft2 else float("nan")
        rdii_in_avg = tot_rdii_ft3 / tot_area_ft2 * IN_PER_FT if tot_area_ft2 else float("nan")
        capture_tot = tot_rdii_ft3 / tot_rain_ft3 if tot_rain_ft3 > 0 else float("nan")

        over_unity = [(n, st["sum_r"]) for n, st in stats_list
                      if math.isfinite(st["sum_r"]) and st["sum_r"] > 1.0 + 1e-9]
        over_capture = []
        for n, st in stats_list:
            rain_ft3 = st["rain_in"] / IN_PER_FT * st["area_acres"] * SQFT_PER_ACRE
            if rain_ft3 > 0 and math.isfinite(st["sum_r"]):
                cap = st["rdii_ft3"] / rain_ft3
                if cap > st["sum_r"] * 1.02 + 1e-6:
                    over_capture.append((n, cap, st["sum_r"]))
        finite_r = [(st["sum_r"], n) for n, st in stats_list
                    if math.isfinite(st["sum_r"])]
        max_r, max_r_sc = max(finite_r) if finite_r else (float("nan"), "-")

        print("Runoff continuity (RDII vs rainfall, sim window, "
              f"{tot_area_ft2 / SQFT_PER_ACRE:,.0f} ac modeled):")
        print(f"    Rainfall volume : {tot_rain_ft3 * GAL_PER_FT3 / 1e6:12,.2f} Mgal"
              f"  ({rain_in_avg:6.2f} in)")
        print(f"    RDII volume     : {tot_rdii_ft3 * GAL_PER_FT3 / 1e6:12,.2f} Mgal"
              f"  ({rdii_in_avg:6.2f} in;  capture = {capture_tot:.4f} of rainfall)")
        print(f"    max sum(R)      : {max_r:.4f}  ({max_r_sc})"
              + ("  - all SCs <= 1.0, no mass generation" if not over_unity else ""))
        if over_unity:
            print(f"    *** {len(over_unity)} SC(s) with R_1+R_2+R_3 > 1 "
                  f"(MASS GENERATION):")
            for n, r in sorted(over_unity, key=lambda x: -x[1]):
                print(f"        {n}: sum(R) = {r:.4f}")
        if over_capture:
            print(f"    *** {len(over_capture)} SC(s) where RDII capture exceeds "
                  f"sum(R) (check forcing/params):")
            for n, cap, r in over_capture:
                print(f"        {n}: capture = {cap:.4f} > sum(R) = {r:.4f}")

    # ------------------------------------------------------------------
    # 4. Sum per junction.
    # ------------------------------------------------------------------
    junction_rdii: dict[str, pd.Series] = {}
    for sc_row in m_obj._sc_df.itertuples():
        if sc_row.name not in sc_rdii:
            continue
        jn = sc_row.outlet
        s  = sc_rdii[sc_row.name]
        junction_rdii[jn] = junction_rdii[jn].add(s, fill_value=0.0) if jn in junction_rdii else s.copy()

    # ------------------------------------------------------------------
    # 5. Write a SWMM5 inflows interface file (all junctions, one file).
    #    Format per PCSWMM/EPA SWMM5 spec:
    #      Line 1: "SWMM5"
    #      Line 2: description
    #      Line 3: routing timestep in seconds (integer)
    #      Line 4: number of variables (1 = FLOW only)
    #      Line 5: "FLOW <unit>"
    #      Line 6: number of nodes
    #      Lines 7..N: node names (one per line)
    #      Blank column-header line
    #      Data rows: <node> <year> <mon> <day> <hr> <min> <sec> <flow>
    #    All nodes are written at each timestep in the same order as the header.
    # ------------------------------------------------------------------
    inp_stem = os.path.splitext(os.path.basename(inp_path))[0]
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(inp_path), "sparsehydro")
    os.makedirs(out_dir, exist_ok=True)

    abs_out_dir  = os.path.abspath(out_dir)
    interface_path = os.path.join(abs_out_dir, f"{inp_stem}_inflows.txt")

    # Build sorted node list and clip all series to the full SWMM window.
    nodes_sorted = sorted(junction_rdii.keys())
    clipped: dict[str, pd.Series] = {}
    for jn in nodes_sorted:
        s = junction_rdii[jn]
        clipped[jn] = s[(s.index >= sim_start) & (s.index <= sim_end)]

    # Linearly interpolate each junction's daily RDII to inflow_dt resolution.
    # Daily values represent average flow over the day; linear interpolation
    # is consistent with how SWMM's own RDII objects feed inflows to the
    # hydraulic solver.  Values are clamped to >= 0 to prevent negative flow
    # from interpolation near zero-flow days.
    inflow_dt_s = int(inflow_dt.total_seconds())
    if inflow_dt < pd.Timedelta(days=1):
        fine_index = pd.date_range(start=sim_start, end=sim_end, freq=inflow_dt)
        fine: dict[str, pd.Series] = {}
        for jn in nodes_sorted:
            s = clipped[jn]
            # Reindex onto a daily grid anchored to sim_start so the daily
            # value sits at the start of each day, then interpolate linearly
            # onto the fine grid and clamp.
            s_daily = s.reindex(
                pd.date_range(start=sim_start, end=sim_end, freq="D"),
                fill_value=0.0,
            )
            s_fine = s_daily.reindex(s_daily.index.union(fine_index)) \
                             .interpolate(method="time") \
                             .reindex(fine_index) \
                             .clip(lower=0.0)
            fine[jn] = s_fine
        all_times = list(fine_index)
        data = fine
    else:
        all_times = sorted(set().union(*[s.index for s in clipped.values()]))
        data = clipped

    # Vectorized write: align all node series onto one time axis, then emit
    # rows in large chunks.  The per-value Series lookup + per-row write()
    # this replaces dominated wall-clock time between the RDII prediction
    # and the SWMM engine start on fine-timestep multi-year runs.
    _t0 = time.time()
    ts_index = pd.DatetimeIndex(all_times)
    mat = (pd.DataFrame({jn: data[jn] for jn in nodes_sorted})
           .reindex(ts_index).fillna(0.0).to_numpy())

    with open(interface_path, "w", encoding="utf-8") as f:
        f.write("SWMM5\n")
        f.write(f"sparsehydro RDII inflows — {inp_stem}\n")
        f.write(f"{inflow_dt_s}\n")     # timestep in seconds
        f.write("1\n")                  # one variable (FLOW)
        f.write(f"FLOW {flow_unit}\n")
        f.write(f"{len(nodes_sorted)}\n")
        for jn in nodes_sorted:
            f.write(f"{jn}\n")
        f.write("Node Year Mon Day Hr Min Sec Flow\n")
        buf: list[str] = []
        for i, t in enumerate(ts_index):
            t_str = (f"{t.year} {t.month:02d} {t.day:02d} "
                     f"{t.hour:02d} {t.minute:02d} {t.second:02d}")
            row = mat[i]
            buf.extend(f"{jn}  {t_str}  {row[j]:.4f}"
                       for j, jn in enumerate(nodes_sorted))
            if len(buf) >= 200_000:
                f.write("\n".join(buf))
                f.write("\n")
                buf.clear()
        if buf:
            f.write("\n".join(buf))
            f.write("\n")

    print(f"Inflows     : {len(nodes_sorted)} junctions @ {inflow_dt_label} "
          f"-> {os.path.basename(interface_path)}  ({time.time() - _t0:.1f}s)")

    # ------------------------------------------------------------------
    # 5b. Write a PCSWMM TSF holding each subcatchment's RDII separately.
    #     The interface file above carries only junction TOTALS, so where
    #     several subcatchments drain to one junction their individual
    #     contributions are not recoverable from it.  This file keeps them
    #     as separate columns.
    #
    #     Written at the model's native DAILY resolution — the sub-daily
    #     values in the interface file are linear interpolation of exactly
    #     these numbers, so nothing is lost and the file stays small
    #     enough to open.
    #
    #     Format is the tab-delimited PCSWMM layout that load_tsf() reads:
    #       IDs:<TAB><sc1><TAB><sc2>...
    #       Date/Time<TAB>Flow<TAB>Flow...
    #       M/d/yyyy<TAB>cfs<TAB>cfs...
    #       M/d/yyyy h:mm:ss AM/PM<TAB><v1><TAB><v2>...
    # ------------------------------------------------------------------
    if write_sc_tsf and sc_rdii:
        _t0 = time.time()
        if sc_tsf_path is None:
            sc_tsf_path = os.path.join(abs_out_dir, f"{inp_stem}_sparsehydro.tsf")
        n_sc = write_series_tsf(sc_tsf_path, sc_rdii, sim_start, sim_end,
                                unit=flow_unit)
        print(f"SC TSF      : {n_sc} subcatchments @ daily "
              f"-> {os.path.basename(sc_tsf_path)}  ({time.time() - _t0:.1f}s)")

    # ------------------------------------------------------------------
    # 6. Patch OPTIONS to reference the interface file; write patched INP.
    #    SWMM reads the interface file via FILES section, not [INFLOWS].
    # ------------------------------------------------------------------
    sim_start_str = f"{sim_start.month}/{sim_start.day}/{sim_start.year}"
    sim_end_str   = f"{sim_end.month}/{sim_end.day}/{sim_end.year}"

    if out_inp is None:
        out_inp = os.path.join(os.path.dirname(inp_path), f"{inp_stem}_sparsehydro.inp")

    # Single pass: routing-only rewrite + [FILES] interface reference in one
    # read/write of the (potentially huge) INP.
    _t0 = time.time()
    _build_routing_only_inp(
        inp_path, out_inp, sim_start_str, sim_end_str,
        sh_nodes=set(junction_rdii.keys()),
        files_content=f'USE INFLOWS   "{interface_path}"\n',
    )

    print(f"Patched INP : {os.path.basename(out_inp)}  ({time.time() - _t0:.1f}s)")

    # ------------------------------------------------------------------
    # 7. Run SWMM — single call, dates driven by the patched INP.
    # ------------------------------------------------------------------
    # openswmm's run_solver ignores its out_file argument and always writes
    # <inp_stem>.out next to the INP it runs, so run against that default
    # path and move the result afterwards if the caller asked for another.
    engine_out = os.path.splitext(out_inp)[0] + ".out"
    out_file = out_file or engine_out
    rpt_file = os.path.splitext(out_inp)[0] + ".rpt"

    total_days = max((sim_end - sim_start) / pd.Timedelta(days=1), 1e-9)
    if engine_dll:
        engine_label = os.path.basename(engine_dll)
    else:
        engine_label = ("openswmm v6 (beta)" if engine == "v6"
                        else "openswmm legacy (SWMM 5.3)")
        if out_version:
            engine_label += f", output tagged as SWMM {out_version}"
    print(f"Engine      : {engine_label}")

    def _run_engine(progress_cb=None) -> int:
        if engine_dll is not None:
            return _run_swmm_dll(engine_dll, out_inp, rpt_file, engine_out,
                                 progress_callback=progress_cb,
                                 total_days=total_days)
        if engine == "v6":
            return _run_v6(out_inp, rpt_file, engine_out, callback=progress_cb)
        if progress_cb is not None:
            return run_solver(out_inp, rpt_file, engine_out,
                              swmm_progress_callback=progress_cb)
        return run_solver(out_inp, rpt_file, engine_out)

    if show_progress:
        try:
            # Plain-text tqdm (not tqdm.auto) — see note at the other tqdm
            # import: VS Code's ipywidgets renderer is unreliable.
            from tqdm import tqdm as _tqdm
        except ImportError:
            _tqdm = None

        if _tqdm is not None:
            with _tqdm(total=100, desc="SWMM routing", unit="%",
                       bar_format="{l_bar}{bar}| {n:.0f}/{total}% [{elapsed}<{remaining}]") as pbar:
                _last = [0.0]
                def _progress(pct: float) -> None:
                    delta = pct * 100 - _last[0]
                    if delta > 0:
                        pbar.update(delta)
                        _last[0] = pct * 100
                err = _run_engine(_progress)
        else:
            def _progress(pct: float) -> None:
                print(f"\r  {pct*100:5.1f}%", end="", flush=True)
            err = _run_engine(_progress)
            print()
    else:
        err = _run_engine()
    if err:
        raise RuntimeError(
            f"SWMM run failed (error code {err}). Check report: {rpt_file}"
        )

    if out_version and engine_dll is None:
        _retag_out_version(engine_out, out_version)

    if in_place:
        final_out = _adopt_outputs_in_place(
            inp_path, engine_out, rpt_file, out_inp, interface_path, abs_out_dir,
        )
        return inp_path, final_out

    if os.path.abspath(out_file) != os.path.abspath(engine_out):
        try:
            os.replace(engine_out, out_file)
        except PermissionError:
            # Destination may be held open (e.g. PCSWMM has it loaded);
            # fall back to copying over it in place.
            import shutil
            shutil.copyfile(engine_out, out_file)
            os.remove(engine_out)
        _reset_creation_time(out_file)
    print(f"Output      : {out_file}")

    return out_inp, out_file


def _adopt_outputs_in_place(
    inp_path: str,
    engine_out: str,
    engine_rpt: str,
    patched_inp: str,
    interface_path: str,
    out_dir: str,
) -> str:
    """
    Adopt a finished sparsehydro run's outputs under the ORIGINAL model stem.

    Moves the engine ``.out``/``.rpt`` to ``<inp_stem>.out`` / ``<inp_stem>.rpt``
    next to *inp_path* (overwriting any previous results, with creation time
    reset so PCSWMM registers them), then deletes the run scaffolding: the
    patched INP and the inflows interface file (plus *out_dir* when that
    leaves it empty).  Returns the adopted ``.out`` path.
    """
    import shutil

    orig_stem = os.path.splitext(inp_path)[0]
    adopted = {}
    for src, dst in ((engine_out, orig_stem + ".out"),
                     (engine_rpt, orig_stem + ".rpt")):
        if not os.path.exists(src):
            continue
        try:
            os.replace(src, dst)
        except PermissionError:
            # Destination may be held open (e.g. PCSWMM has results loaded);
            # fall back to copying over it in place.
            shutil.copyfile(src, dst)
            os.remove(src)
        _reset_creation_time(dst)
        adopted[os.path.splitext(dst)[1]] = dst

    for scaffold in (patched_inp, interface_path):
        try:
            os.remove(scaffold)
        except OSError:
            pass
    try:
        os.rmdir(out_dir)          # only succeeds when empty
    except OSError:
        pass

    print(f"Output      : {adopted.get('.out', '(no .out produced)')}  "
          f"[adopted in place; temp INP + inflows file removed]")
    return adopted.get(".out", orig_stem + ".out")


# ---------------------------------------------------------------------------
# Future-condition imperviousness rescaling
# ---------------------------------------------------------------------------

def rescale_imperv(
    bundle: dict,
    new_imperv_pct: float,
    new_mean_perv: Optional[float] = None,
) -> None:
    """
    Adjust ia_max in a loaded pkl bundle for a new %imperviousness value.

    The ia_max currently in the bundle was calibrated as:
        ia_max_calibrated = ia_max_base * imp_multiplier
    where ``imp_multiplier = (1 - imperv_pct/100) / mean_perv``.

    This function reverses that scaling and applies a new one based on
    *new_imperv_pct* and *new_mean_perv*.  Modifies the bundle in place:
    for current bundles the ``_ia_max`` values inside
    ``bundle["params_v2"]["params"]`` are rewritten; for legacy bundles
    (pre-``params_v2``) the pickled ``bundle["model"]`` object is mutated.

    Use cases: land-use change scenarios (a subcatchment's imperviousness
    changed after calibration), or transferring a donor SC's bundle to an
    unmetered subcatchment with different imperviousness.

    Parameters
    ----------
    bundle : dict
        Bundle loaded by ``CalOptimizer.load_sc_model()``.
    new_imperv_pct : float
        New percent imperviousness (0–100).
    new_mean_perv : float or None
        New area-weighted mean pervious fraction across the watershed.
        If ``None``, uses the calibration-time mean stored in the bundle.

    Raises
    ------
    ValueError
        If the bundle does not contain imperviousness scaling information
        (i.e. ``imp_multiplier`` is ``None`` — imperv scaling was off).
    """
    imp_mult = bundle.get("imp_multiplier")
    mean_perv_cal = bundle.get("mean_perv")

    if imp_mult is None or imp_mult == 0.0:
        raise ValueError(
            "This bundle was not calibrated with imperviousness scaling "
            "(imp_multiplier is None or 0). Cannot rescale."
        )
    if mean_perv_cal is None or mean_perv_cal <= 0:
        raise ValueError(
            "Bundle is missing a valid mean_perv value. Cannot rescale."
        )

    effective_mean_perv = float(new_mean_perv) if new_mean_perv is not None else mean_perv_cal
    new_perv = max(0.0, 1.0 - new_imperv_pct / 100.0)
    new_multiplier = new_perv / effective_mean_perv if effective_mean_perv > 0 else 1.0

    pv2 = bundle.get("params_v2")
    if pv2 is not None:
        # Current bundles: rescale the plain parameter dict directly.
        params = pv2["params"]
        for name in params:
            if name.endswith("_ia_max"):
                params[name] = params[name] / imp_mult * new_multiplier
    else:
        # Legacy bundles: mutate the pickled model object in place.
        m = bundle["model"]
        try:
            param_meta = m.param_meta  # list of (name, lb, ub, log)
        except AttributeError:
            param_meta = getattr(m, "_param_meta", [])

        for name, *_ in param_meta:
            if "ia_max" not in name:
                continue
            try:
                current_val = m.get_parameter_value(name)
            except AttributeError:
                continue
            base = current_val / imp_mult
            try:
                m.set_parameter(name, base * new_multiplier)
            except AttributeError:
                pass

    # Update bundle metadata to reflect the new imperv state.
    bundle["imperv_pct"] = new_imperv_pct
    bundle["imp_multiplier"] = new_multiplier
    if new_mean_perv is not None:
        bundle["mean_perv"] = effective_mean_perv


def rescale_bfi(
    bundle: dict,
    new_bfi: float,
    new_mean_bfi: Optional[float] = None,
) -> None:
    """
    Adjust R_3 (the slow/groundwater response fraction) in a loaded pkl
    bundle for a new BFI value.

    The R_3 currently in the bundle was calibrated as:
        R_3_calibrated = R_3_watershed * r3_multiplier
    where ``r3_multiplier = bfi / mean_bfi`` (this SC's BFI relative to the
    metershed's area-weighted mean BFI — see ``_prepare_objective_data``).

    This function reverses that scaling and applies a new one based on
    *new_bfi* and *new_mean_bfi*.  Modifies the ``_R_3`` value inside
    ``bundle["params_v2"]["params"]`` in place.

    Use cases: transferring a donor SC's bundle to an unmetered
    subcatchment with a different BFI, or updating a bundle after the BFI
    estimate for a subcatchment is revised.

    Bundles written before ``mean_bfi``/``r3_multiplier`` were added to
    ``save_sc_models()`` lack the required metadata — backfill them first
    with ``scripts/patch_pkl_bfi.py``.

    Parameters
    ----------
    bundle : dict
        Bundle loaded by ``CalOptimizer.load_sc_model()``.
    new_bfi : float
        New baseflow index for this subcatchment.
    new_mean_bfi : float or None
        New area-weighted mean BFI across the watershed.  If ``None``,
        uses the calibration-time mean stored in the bundle.

    Raises
    ------
    ValueError
        If the bundle lacks ``r3_multiplier``/``mean_bfi`` metadata (run
        the backfill script), has ``r3_multiplier == 0`` (scaling cannot
        be reversed), or has no ``params_v2`` recipe (legacy bundle —
        re-save it with a current ``save_sc_models()``).
    """
    r3_mult = bundle.get("r3_multiplier")
    mean_bfi_cal = bundle.get("mean_bfi")

    if r3_mult is None or mean_bfi_cal is None:
        raise ValueError(
            "Bundle is missing r3_multiplier/mean_bfi metadata. It was "
            "written before BFI rescaling support — backfill it with "
            "scripts/patch_pkl_bfi.py, then retry."
        )
    if r3_mult == 0.0:
        raise ValueError(
            "Bundle has r3_multiplier == 0 (calibrated BFI was 0); the "
            "R_3 scaling cannot be reversed."
        )
    if mean_bfi_cal <= 0:
        raise ValueError("Bundle has an invalid mean_bfi value. Cannot rescale.")

    pv2 = bundle.get("params_v2")
    if pv2 is None:
        raise ValueError(
            "Bundle has no params_v2 recipe (legacy bundle). Re-save it "
            "with a current save_sc_models() before BFI rescaling."
        )

    effective_mean_bfi = float(new_mean_bfi) if new_mean_bfi is not None else mean_bfi_cal
    if effective_mean_bfi <= 0:
        raise ValueError("new_mean_bfi must be positive.")
    new_multiplier = float(new_bfi) / effective_mean_bfi

    # R_3 is the last of the alphabetically-sorted R parameter names —
    # the same convention save_sc_models() uses (self._r_sorted[-1]).
    params = pv2["params"]
    r_names = sorted(n for n in params if "_R_" in n)
    if not r_names:
        raise ValueError("Bundle params_v2 contains no R parameters.")
    r3_name = r_names[-1]

    new_r3 = params[r3_name] / r3_mult * new_multiplier

    # Mass-conservation guard: the R values are fractions of rainfall
    # entering the system, so their sum must not exceed 1 — a rescaled R_3
    # that pushes sum(R) past 1 would GENERATE mass.  Check within the same
    # RTK set as R_3 (keys sharing its prefix before '_R_').
    prefix = r3_name.split("_R_")[0]
    other_r = sum(v for k, v in params.items()
                  if "_R_" in k and k != r3_name
                  and k.split("_R_")[0] == prefix)
    total_r = other_r + new_r3
    if total_r > 1.0 + 1e-9:
        old_r3 = params[r3_name]
        raise ValueError(
            f"BFI rescale rejected for {bundle.get('sc_name', '?')!r}: "
            f"R_1+R_2+R_3 would exceed 1 (mass generation).\n"
            f"  new_bfi={float(new_bfi):g} / mean_bfi={effective_mean_bfi:g} "
            f"-> {r3_name} {old_r3:.6g} -> {new_r3:.6g}; "
            f"sum(R) = {other_r:.6g} + {new_r3:.6g} = {total_r:.6g} > 1\n"
            f"This subcatchment's BFI is too high relative to the donor "
            f"metershed's calibrated R_3. Use a donor metershed with a "
            f"higher mean BFI or a lower watershed R_3, or verify the "
            f"BFI attribute value."
        )

    params[r3_name] = new_r3

    # Update bundle metadata to reflect the new BFI state.
    bundle["bfi"] = float(new_bfi)
    bundle["r3_multiplier"] = new_multiplier
    if new_mean_bfi is not None:
        bundle["mean_bfi"] = effective_mean_bfi
