# Kaggle

~7+ hours maxvit_agg test with full TTA which is almost max allowed time of 9 hours.
~1+ hour of it is preprocessing and ~6+ hours is inference.

# Local machine performance

Fold 4 (batch size 8 for maxvit & convnext and 3 for maxvit_agg) validation in seconds:
    no TTA:
        maxvit: 96
        maxvit_agg: 162
        convnext: 91
    rotate 90 (0, 1, 2, 3) TTA:
        maxvit: 101
        maxvit_agg: ~330
        convnext: 92
    hflips + vflips + rotate 90 (0, 1, 3) TTA:
        maxvit_agg: ~750
    vflips + rotate 90 (0, 1, 2, 3) TTA:
        maxvit_agg: ~660
    full TTA:
        maxvit: 104
        maxvit_agg: ~950
        convnext: 106

- Full TTA each model: (950 + 106 + 103) / 950 * 7 + 1 ~ 9.5 hours
- Full TTA maxvit and convnext + hflips & vflips & rotate 90 (0, 1, 3) TTA maxvit_agg v1: (750 + 106 + 103) / 950 * 7 + 1 ~ 8 hours
- Full TTA maxvit and convnext + no TTA maxvit_agg v1 + vflips & rotate 90 (0, 1, 2, 3) TTA maxvit_agg v2: (162 + 660 + 106 + 103) / 950 * 7 + 1 ~ 8.6 hours