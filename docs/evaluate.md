# Evaluation

## Human Motion Imitation

### Youtube-Dancer-18
```bash
python scripts/evaluate/eval_imitator.py --gpu_ids 1 \
    --eval_datasets  "Youtube-Dancer-18-Tiny?=/p300/tpami/datasets_round-1/Youtube-Dancer-18" \
    --output_dir  "/p300/tpami/ablationStudy/WarpingStrategies-round-2/evaluations" \
    --num_source 8  \
    --src_path  ""  \
    --ref_path  ""  
```


