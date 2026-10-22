BEGIN;

INSERT INTO reid_models (
    model_key, backend, object_class, model_version, weights_uri, config_uri,
    feature_dim, input_height, input_width, distance_metric, l2_normalized
)
VALUES
    (
        'msmt17_sbs_r50_ibn', 'fastreid', 'person', 'v0.1.1',
        'models/fastreid/msmt17_sbs_R50-IBN.pth',
        'models/fastreid/configs/msmt17_sbs_R50-IBN.yml',
        2048, 256, 128, 'cosine', TRUE
    ),
    (
        'veri_sbs_r50_ibn', 'fastreid', 'vehicle', 'v0.1.1',
        'models/fastreid/veri_sbs_R50-IBN.pth',
        'models/fastreid/configs/veri_sbs_R50-IBN.yml',
        2048, 256, 256, 'cosine', TRUE
    )
ON CONFLICT (model_key, model_version) DO UPDATE SET
    weights_uri = EXCLUDED.weights_uri,
    config_uri = EXCLUDED.config_uri,
    active = TRUE;

INSERT INTO reid_threshold_policies (model_id, candidate_top_k, review_threshold, confirm_threshold, min_score_margin)
SELECT model_id, 20, 0.65, 0.75, 0.05
FROM reid_models
WHERE model_key IN ('msmt17_sbs_r50_ibn', 'veri_sbs_r50_ibn')
  AND NOT EXISTS (
      SELECT 1 FROM reid_threshold_policies p
      WHERE p.model_id = reid_models.model_id
        AND p.camera_from_id IS NULL
        AND p.camera_to_id IS NULL
  );

COMMIT;
