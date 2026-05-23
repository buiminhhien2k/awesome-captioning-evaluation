# Activate venv (optional but cleaner)
& "venv/Scripts/Activate.ps1"

# python compute_all_metric.py --dataset polaris --metrics_name standard --clip_model "ViT-B/32"
#
# python compute_all_metric.py --dataset polaris --metrics_name bert-score --clip_model "ViT-L/14"
# python compute_all_metric.py --dataset polaris --metrics_name bert-score++ --clip_model "ViT-L/14"
# python compute_all_metric.py --dataset polaris --metrics_name blip2-score --clip_model "ViT-L/14"
# python compute_all_metric.py --dataset polaris --metrics_name umic-score --clip_model "ViT-L/14"
# python compute_all_metric.py --dataset flickrExpert --metrics_name clip-score --clip_model "ViT-B/32"
# python compute_all_metric.py --dataset flickrExpert --metrics_name pac-score --clip_model "ViT-B/32"
# python compute_all_metric.py --dataset flickrExpert --metrics_name pac-score++ --clip_model "ViT-B/32"

# python compute_all_metric.py --dataset flickr8k --metrics_name polos --clip_model "ViT-L/14"
python compute_all_metric.py --dataset flickr8k --metrics_name blip2-score --clip_model "ViT-L/14"
# python compute_all_metric.py --dataset flickrExpert --metrics_name umic-score --clip_model "ViT-L/14"
# python compute_all_metric.py --dataset flickrExpert --metrics_name pac-score --clip_model "ViT-L/14"
# python compute_all_metric.py --dataset flickrExpert --metrics_name pac-score++ --clip_model "ViT-L/14"

# python compute_all_metric.py --dataset composite --metrics_name mid-score --clip_model "ViT-B/32"
# python compute_all_metric.py --dataset polaris --metrics_name mid-score --clip_model "ViT-B/32"
# python compute_all_metric.py --dataset polaris --metrics_name polos --clip_model "ViT-L/14"

# python compute_all_metric.py --dataset polaris --metrics_name clip-score --clip_model "ViT-B/32"
# python compute_all_metric.py --dataset polaris --metrics_name pac-score --clip_model "ViT-B/32"
# python compute_all_metric.py --dataset polaris --metrics_name pac-score++ --clip_model "ViT-B/32"
#
# python compute_all_metric.py --dataset polaris --metrics_name clip-score --clip_model "ViT-L/14"
# python compute_all_metric.py --dataset polaris --metrics_name pac-score --clip_model "ViT-L/14"
# python compute_all_metric.py --dataset polaris --metrics_name pac-score++ --clip_model "ViT-L/14"

Read-Host "All jobs finished. Press ENTER to exit"