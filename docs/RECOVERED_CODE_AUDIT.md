# Audit of the recovered release

Recovered from the DOI already present in the supplied manuscript, not from an inferred unrelated project.
- Concept record: 10.5281/zenodo.22730609.
- Retrieved API record identifies version record 22730610 and archive `manofseven7/HetST-GNN-v1.0.0.zip`.
- Repository commit embedded in ZIP: 45830aa4fc7cbfa27f804352e7ea21477147cc3e.
- The archive contains implementation files and a CSV reproducing manuscript table entries. It does not contain per-seed predictions, checkpoints, or raw evidence supporting the old reported means/SDs. This observation does not establish how the original values were obtained.

## Confirmed implementation defects

1. `hetstgnn/simulator.py`: a shipment is scheduled to a downstream node without decrementing any upstream inventory. A three-node fixture with initial total inventory12 and customer service2 ends with total inventory12, although there are no external root replenishments. Two units have been created. Multiple children may also independently consume the same nominal upstream availability.
2. `_finalize` in `hetstgnn/data.py` computes demand mean and SD on the complete demand matrix before temporal splitting. These enter lead times and base-stock parameters. The resulting benchmark construction depends on future demand.
3. UCI product/country selection also uses complete-period volume. The adapter topology has two echelons; it does not implement the published five-echelon/180-node UCI description.
4. `run_ebs` uses contemporaneous (already standardized) features at target t, while neural/tree models use windows ending t-1. This is not an equal-information, two-cycle forecasting comparison, and physical inventory arithmetic on standardized variables is inappropriate.
5. `load_problem` passes the full edge list, including reverse information edges, into the physical-flow simulator.
6. Neural and tabular targets use t while their most recent input is t-1; this is a one-cycle rather than the two-cycle manuscript horizon.
7. Existing CSV values are not recalculated by figure generation. Figure regeneration is not experiment reproduction.

## New reconstruction

The new `hetst/` implementation is independently written. It conserves physical material; separates transfer edges from information edges; fits construction/scaling on the training prefix; uses horizon2 for all methods; saves per-run forecasts, models, training curves and decision ledgers; and reports only newly executed results. Because the data construction and architecture budget are explicit and different from the original undocumented experiments, the new measurements are not claimed to replicate their numbers.

The old release remains unmodified in the supplied recovery archive for provenance. Do not use it to generate submission evidence without repairing these defects.
