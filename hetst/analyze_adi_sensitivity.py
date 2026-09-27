"""Apply alternative ADI cutoffs to recorded gated and no-gate predictions.

The experts are not refitted. Each cutoff chooses either the already selected
relational correction or exact local fallback, matching the role of the ADI
gate in ``fit_igrb``.
"""
from pathlib import Path

import pandas as pd


ROOT = Path("results")
OUT = ROOT / "igrb_analysis" / "adi_sensitivity.csv"
CUTOFFS = (1.00, 1.20, 1.32, 1.50, 2.00)
CASES = ("seasonal", "correlated", "fmcg", "retail", "intermittent")
METRICS = ("f1", "auprc", "brier", "rmse_pos")


def main() -> None:
    final = pd.read_csv(ROOT / "igrb_final" / "metrics.csv").query("shift == 'none'")
    fmcg = pd.read_csv(ROOT / "igrb_fmcg" / "metrics.csv").query("shift == 'none'")
    no_adi = pd.read_csv(ROOT / "igrb_noadi" / "metrics.csv").query("shift == 'none'")

    local = pd.concat([final, fmcg], ignore_index=True).query("method == 'LocalXGBTuned'")
    relational = final.query("method == 'IGRB'").copy()
    # In sparse cases the final run is exact fallback; use the recorded no-gate
    # run to recover the already fitted relational correction.
    sparse_relational = no_adi.query("method == 'IGRB'")
    relational = pd.concat(
        [relational.query("case not in ['retail', 'intermittent']"),
         fmcg.query("method == 'IGRB'"), sparse_relational],
        ignore_index=True,
    )

    rows = []
    for case in CASES:
        local_case = local.query("case == @case").set_index("seed")
        graph_case = relational.query("case == @case").set_index("seed")
        if not local_case.index.equals(graph_case.index):
            raise RuntimeError(f"Seed mismatch for {case}")
        adi = float(local_case.training_adi.mean())
        for cutoff in CUTOFFS:
            active = adi <= cutoff
            chosen = graph_case if active else local_case
            row = {
                "adi_cutoff": cutoff,
                "case": case,
                "training_adi": adi,
                "graph_active": active,
                "n_model_seeds": len(chosen),
            }
            for metric in METRICS:
                row[metric] = float(chosen[metric].mean())
                row[f"local_{metric}"] = float(local_case[metric].mean())
                row[f"delta_{metric}"] = row[metric] - row[f"local_{metric}"]
            rows.append(row)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(OUT)


if __name__ == "__main__":
    main()
