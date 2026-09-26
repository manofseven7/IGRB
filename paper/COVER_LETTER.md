Dear Editor-in-Chief,

Please consider our manuscript, “Intermittency-aware graph residual boosting for selective back-order prediction in multi-echelon supply chains,” for publication as a Research Article in Expert Systems with Applications.

The manuscript addresses a practical weakness of graph-based supply-chain forecasting: network context is not uniformly informative and may degrade predictions under sparse demand. We introduce intermittency-aware graph residual boosting (IGRB), which combines a strong local boosted expert with causal parent, child, sibling and network-level context. A chronological validation procedure controls the graph correction, and an average inter-demand interval screen can return exactly to the local expert. A separate validation-cost guardrail prevents forecast improvements from automatically triggering emergency purchases.

The evaluation uses a material-conserving multi-echelon simulator, three synthetic demand regimes, retail-derived traces, two operational shifts, five optimization seeds and up to ten independent demand replications. Across ten unchanged correlated-demand records, IGRB improves macro F1 by 0.0269 over its tuned local expert (paired 95% bootstrap interval 0.0080--0.0436; nine wins). Seasonal evidence is mixed after confirmatory records are included, and the method deliberately ties the local expert under intermittent demand. Guarded closed-loop mean cost is lower in correlated demand under all three evaluated conditions. We explicitly state that operational states and shortage labels are simulated and do not claim universal graph-model or inventory-policy superiority.

The work fits the journal’s focus on the design, testing and management of intelligent systems applied to industrial problems. Its contribution is an auditable selective-expert design, accompanied by strong tabular and graph baselines, component ablations, independent-replication uncertainty, frozen-model stress tests and executable artifacts.

This manuscript is original, is not under consideration elsewhere, and has been approved by all authors. The authors declare no competing interests. [Please replace this sentence if any statement differs.] The manuscript is blinded; complete author information, affiliations, ORCID identifiers and CRediT contributions will be supplied separately.

Sincerely,

[Corresponding author name]
[Affiliation]
[Email]
