# Multi-Factor Fusion Enabled Federated POI Recommendation

This directory contains the POI recommendation component used to reproduce and extend the paper:

> Multi-Factor Fusion Enabled Federated POI Recommendation in Personal Data Trusteeship Scenario

The original repository was a collection of classic POI recommendation baselines. We keep those baselines intact and add a paper-oriented implementation entry point, `prog_fedmff.py`, for the proposed federated multi-factor fusion model.

## Paper-Oriented Method

The proposed model follows the paper's three-party personal data trusteeship setting:

- Recommendation service provider: maintains and aggregates the global model.
- Data trustees: hold local user check-in records and perform local training.
- Users: provide check-in histories through trustees without direct raw-data exchange between trustees.

The implementation in `prog_fedmff.py` covers the main modeling pipeline:

- trustee-level federated training with FedAvg aggregation
- user-POI sequence modeling from local check-in histories
- similar-user graph built from interaction overlap
- similar-POI graph built from co-visiting users
- optional social graph input, with similarity-neighbor fallback when social links are unavailable
- graph attention aggregation for similar-user, social-relation, and similar-POI factors
- GRU-based long/short-term behavior encoder with temporal attention
- learnable user-attribute representation placeholder
- semantic-level attention fusion over five factors:
  - similar users
  - social relationships
  - user attributes
  - spatiotemporal behavior
  - similar POIs
- BPR optimization for top-k POI recommendation
- per-user metric export for statistical significance testing

## Added Files

```text
prog_fedmff.py          # Paper-style federated multi-factor fusion model
prog_mostpop.py         # MostPop non-personalized baseline
significance_test.py    # User-level significance testing and bootstrap CI
requirements_paper.txt  # PyTorch dependency for the new model
IMPLEMENTATION_STATUS.md
```

Legacy baseline files are still available:

```text
prog_prme.py
prog_poi2vec.py
prog_geoie.py
prog_fpmc_lr.py
prog_bpr_gru_spatial.py
```

## Datasets

The paper uses:

- Foursquare-NYC
- Foursquare-TKY
- Yelp

Prepared local sequence files currently include:

```text
poidata/Foursquare_NYC/sequence/Foursquare_NYC.txt
poidata/Foursquare_TKY/sequence/Foursquare_TKY.txt
poidata/Gowalla/sequence/Gowalla.txt
```

The expected sequence format is:

```text
check_times pois_different u_id u_pois u_times u_coordinates
```

Example field format:

```text
u_pois:        poi1/poi2/poi3
u_times:       t1/t2/t3
u_coordinates: lat1,lng1/lat2,lng2/lat3,lng3
```

Optional social relationship file:

```text
user_a,user_b
u1,u2
u1,u3
```

Optional user attribute file:

```text
u_id,gender,age_group,home_region
u1,male,20-29,NYC
u2,female,30-39,TKY
```

Numeric and categorical attributes are both accepted. Categorical attributes are encoded per column and projected into the user-attribute representation used by the multi-factor fusion module.

Foursquare-NYC/TKY can be prepared from the TSMC2014 dataset:

```powershell
cd C:\Users\86135\PycharmProjects\pythonProject3\re1\FedMER-main\Point-of-Interest-Recommendation-master\poidata

python prepare_tsmc2014_foursquare.py `
  --input Foursquare\dataset_tsmc2014\dataset_TSMC2014_NYC.txt `
  --output Foursquare_NYC\sequence\Foursquare_NYC.txt

python prepare_tsmc2014_foursquare.py `
  --input Foursquare\dataset_tsmc2014\dataset_TSMC2014_TKY.txt `
  --output Foursquare_TKY\sequence\Foursquare_TKY.txt
```

## Installation

The paper-oriented implementation requires PyTorch:

```powershell
cd C:\Users\86135\PycharmProjects\pythonProject3\re1\FedMER-main\Point-of-Interest-Recommendation-master
pip install -r requirements_paper.txt
```

The legacy baselines may require their original older dependencies.

## Run the Proposed Model

Example on Foursquare-NYC:

```powershell
python prog_fedmff.py `
  --file poidata\Foursquare_NYC\sequence\Foursquare_NYC.txt `
  --rounds 20 `
  --num-trustees 5 `
  --local-epochs 5 `
  --gat-layers 2 `
  --embedding-dim 32 `
  --topk 10,20 `
  --per-user-output results\fedmff_nyc_per_user.csv
```

Example on Foursquare-TKY:

```powershell
python prog_fedmff.py `
  --file poidata\Foursquare_TKY\sequence\Foursquare_TKY.txt `
  --rounds 20 `
  --num-trustees 5 `
  --local-epochs 5 `
  --gat-layers 2 `
  --embedding-dim 32 `
  --topk 10,20 `
  --per-user-output results\fedmff_tky_per_user.csv
```

When social links or user attributes are available, add:

```powershell
--social-file data\nyc_social.csv `
--user-attr-file data\nyc_user_attributes.csv
```

The default parameters follow the paper's reported setting where possible:

```text
GAT layers:        2 for Foursquare-NYC/TKY
local epochs:      5 for Foursquare-NYC/TKY
embedding size:    32
metrics:           Precision@10/20, Recall@10/20, MAP, NDCG
```

## Run MostPop Baseline

MostPop ranks POIs by training-set frequency and filters POIs already observed in each user's training history before producing the Top-K list.

```powershell
python prog_mostpop.py `
  --file poidata\Foursquare_NYC\sequence\Foursquare_NYC.txt `
  --topk 5,10,15,20 `
  --per-user-output results\mostpop_nyc_per_user.csv

python prog_mostpop.py `
  --file poidata\Foursquare_TKY\sequence\Foursquare_TKY.txt `
  --topk 5,10,15,20 `
  --per-user-output results\mostpop_tky_per_user.csv
```

The numerical values can be copied into Table 2 in the manuscript. Generated
tables and per-user CSV files should be kept local under `results/` and are not
included in the clean code package.

## Statistical Significance

The model exports per-user metrics so user-level significance tests can be reported. After producing per-user CSV files for the proposed model and a baseline, run:

```powershell
python significance_test.py `
  --ours results\fedmff_nyc_per_user.csv `
  --baseline results\mostpop_nyc_per_user.csv `
  --metrics precision@10,precision@20,recall@10,recall@20,ap@20
```

The script reports:

- mean metric for each method
- mean paired difference
- sign-test p-value
- paired t-test p-value with normal approximation
- 95% bootstrap confidence interval for the paired improvement

## Current Limitations

The implementation matches the paper architecture as closely as the available local data allows, but exact reproduction of the published table still requires:

- Yelp raw data preparation
- explicit social relationship files for all datasets
- real user attribute features
- per-user outputs of all compared baselines, especially FedGNN/PMF/FedNCF, for significance testing

When social links or user attributes are unavailable, `prog_fedmff.py` uses similarity-neighbor fallback and learned user-attribute embeddings. This keeps the model runnable while making the missing data assumptions explicit.
