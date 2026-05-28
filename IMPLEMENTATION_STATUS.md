# Implementation Status

This repository originally contained legacy POI recommendation baselines and a separate FedSAW-style federated-learning wrapper. The paper `main.pdf` proposes a different model: a personal-data-trusteeship federated POI recommendation framework with multi-factor feature representation and semantic attention fusion.

The new implementation keeps the legacy baselines unchanged and adds a paper-oriented implementation in `prog_fedmff.py`.

## Implemented Components

`prog_fedmff.py` implements the following paper-aligned components:

- three-party trusteeship abstraction through trustee-level data partitioning
- local trustee training and service-provider-side FedAvg aggregation
- BPR loss for top-k POI recommendation
- similar-user graph from user check-in overlap
- similar-POI graph from shared visitor sets
- optional explicit social graph input
- graph attention aggregation for:
  - similar-user representation
  - social-relation representation
  - similar-POI representation
- GRU sequence encoder for user behavioral preference
- temporal attention over historical check-ins
- user-attribute representation from an optional attribute CSV, with learnable embedding fallback
- semantic-level attention fusion over:
  - similar users
  - social relationships
  - user attributes
  - spatiotemporal behavior
  - similar POIs
- Top-K evaluation
- per-user metric export for significance testing

Additional files:

- `prog_mostpop.py`: popularity-based non-personalized baseline requested by reviewers, including per-user metric export.
- `significance_test.py`: user-level significance tests and bootstrap confidence intervals.
- `results/README.md`: documents local generated outputs. Concrete result CSV/table files are intentionally not kept in the clean code package.

## Dataset Coverage

Prepared locally:

- Foursquare-NYC
- Foursquare-TKY
- Gowalla, used only for auxiliary baseline validation

Not yet prepared:

- Yelp

## Known Gaps Against Exact Paper Reproduction

The current local sequence files contain check-in sequences and coordinates. They do not contain all auxiliary information assumed by the paper.

Missing or incomplete inputs:

- explicit social relationships for Foursquare-NYC/TKY
- user personal attributes
- Yelp processed sequences
- per-user outputs of published baselines such as FedGNN, PMF, FedNCF, PrivRec, FedMF and FCF

Fallback behavior:

- If explicit social links are absent, the implementation uses similar-user neighbors as a social-relation fallback.
- If user attributes are absent, the implementation uses learnable user-attribute embeddings.
- If user attributes are provided, numeric and categorical columns are encoded and projected into the attribute factor.

Therefore, the code now implements the paper's model structure and training/evaluation workflow, but exact numerical reproduction of the published experimental table still requires the full processed datasets and all baseline outputs used in the paper.
