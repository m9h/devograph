"""Gene regulatory network (GRN) data loaders.

General-purpose GRN loaders are now in ``hgx._grn`` and re-exported here
for backward compatibility.  This module keeps only the Pando-specific
loader ``load_pando_modules``.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from hgx._grn import (  # noqa: F401  -- re-export
    _correlation_modules,
    _greedy_modules,
    grn_to_temporal_hypergraphs,
    load_grn_from_anndata,
    load_grn_from_csv,
    load_grn_from_edge_list,
)
from hgx._hypergraph import Hypergraph


# ---------------------------------------------------------------------------
# Pando-specific loader (stays in devograph)
# ---------------------------------------------------------------------------


def load_pando_modules(
    coef_csv: str | Path,
    modules_csv: str | Path | None = None,
    *,
    padj_threshold: float = 0.05,
    tf_col: str = "tf",
    target_col: str = "target",
    estimate_col: str = "estimate",
    padj_col: str = "padj",
    module_col: str = "module",
    gene_col: str = "gene",
) -> Hypergraph:
    """Load Pando GRN inference output as a Hypergraph.

    Reads a coefficients CSV (as produced by Pando's ``coef()``
    method), filters to significant edges (``padj < padj_threshold``),
    and builds a hypergraph.

    Args:
        coef_csv: Path to the Pando coefficients CSV with columns
            for TF, target, estimate (weight), and adjusted p-value.
        modules_csv: Optional path to a CSV mapping genes to module
            IDs (columns: *gene_col*, *module_col*).  If provided,
            modules define hyperedges.  Otherwise, edges are grouped
            by source TF.
        padj_threshold: Maximum adjusted p-value for inclusion.
        tf_col: Column name for TF in *coef_csv*.
        target_col: Column name for target gene in *coef_csv*.
        estimate_col: Column name for effect estimate (weight).
        padj_col: Column name for adjusted p-value.
        module_col: Column name for module ID in *modules_csv*.
        gene_col: Column name for gene name in *modules_csv*.

    Returns:
        A :class:`~hgx.Hypergraph`.
    """
    coef_csv = Path(coef_csv)
    with open(coef_csv, newline="") as fh:
        sample = fh.read(4096)
        fh.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        reader = csv.DictReader(fh, dialect=dialect)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Pando coefficients CSV is empty: {coef_csv}")

    available = set(rows[0].keys())
    for col_name, label in [
        (tf_col, "tf"),
        (target_col, "target"),
        (estimate_col, "estimate"),
        (padj_col, "padj"),
    ]:
        if col_name not in available:
            raise ValueError(
                f"Column {label}={col_name!r} not found. "
                f"Available: {sorted(available)}"
            )

    # Filter to significant edges
    edges: list[tuple[str, str, float]] = []
    for row in rows:
        padj = float(row[padj_col])
        if padj < padj_threshold:
            tf = row[tf_col].strip()
            target = row[target_col].strip()
            w = float(row[estimate_col])
            edges.append((tf, target, w))

    if not edges:
        raise ValueError(
            f"No significant edges found (padj < {padj_threshold}) "
            f"in {coef_csv}"
        )

    # Load optional modules mapping
    modules: dict[str, list[str]] | None = None
    if modules_csv is not None:
        modules_csv = Path(modules_csv)
        with open(modules_csv, newline="") as fh:
            sample = fh.read(4096)
            fh.seek(0)
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
            reader = csv.DictReader(fh, dialect=dialect)
            mod_rows = list(reader)

        mod_available = set(mod_rows[0].keys()) if mod_rows else set()
        if gene_col not in mod_available or module_col not in mod_available:
            raise ValueError(
                f"Modules CSV must have columns {gene_col!r} and "
                f"{module_col!r}. Found: {sorted(mod_available)}"
            )

        mod_groups: dict[str, list[str]] = defaultdict(list)
        for row in mod_rows:
            mid = row[module_col].strip()
            gene = row[gene_col].strip()
            mod_groups[mid].append(gene)
        modules = dict(mod_groups)

    return load_grn_from_edge_list(edges, modules=modules)
