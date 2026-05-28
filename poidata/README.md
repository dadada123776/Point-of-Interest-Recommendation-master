# Dataset Directory

Large raw and processed datasets are not stored in this code directory.

Local working copies were moved to:

```text
C:\Users\86135\PycharmProjects\pythonProject3\re1\external_datasets\poi_recommendation
```

Expected processed sequence files:

```text
poidata/Foursquare_NYC/sequence/Foursquare_NYC.txt
poidata/Foursquare_TKY/sequence/Foursquare_TKY.txt
poidata/Gowalla/sequence/Gowalla.txt
```

To regenerate Foursquare-NYC/TKY sequence files, place the TSMC2014 raw files
under `poidata/Foursquare/dataset_tsmc2014/` and run:

```powershell
python poidata\prepare_tsmc2014_foursquare.py `
  --input poidata\Foursquare\dataset_tsmc2014\dataset_TSMC2014_NYC.txt `
  --output poidata\Foursquare_NYC\sequence\Foursquare_NYC.txt

python poidata\prepare_tsmc2014_foursquare.py `
  --input poidata\Foursquare\dataset_tsmc2014\dataset_TSMC2014_TKY.txt `
  --output poidata\Foursquare_TKY\sequence\Foursquare_TKY.txt
```

To regenerate Gowalla, place `loc-gowalla_totalCheckins.txt.gz` under
`poidata/Gowalla/` and run:

```powershell
python poidata\prepare_snap_gowalla.py `
  --input poidata\Gowalla\loc-gowalla_totalCheckins.txt.gz `
  --output poidata\Gowalla\sequence\Gowalla.txt
```
