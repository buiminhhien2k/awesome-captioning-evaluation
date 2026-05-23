# -*- coding: utf-8 -*-
import os

import click
import yaml
import torch

from .estimators import PolosEstimator, QualityEstimator
from .model_base import ModelBase
from .ranking import PolosRanker
from models.polos.my_utils.torchnlp_custom import download_file_maybe_extract

str2model = {
    "PolosEstimator": PolosEstimator,
    "PolosRanker": PolosRanker,
    # Model that use source only:
    "QualityEstimator": QualityEstimator,
}

def get_cache_folder():
    # if "HOME" in os.environ:
    # cache_directory = os.environ["HOME"] + "/.cache/torch/yuigawada/"
    cache_directory = "checkpoints/yuigawada/"
    if not os.path.exists(cache_directory):
        os.makedirs(cache_directory)
    return cache_directory
    # else:
    #     raise Exception("HOME environment variable is not defined.")


import os

def download_model(model: str, saving_directory: str = None) -> ModelBase:
    """Function that loads pretrained models from AWS.
    :param model: Name of the model to be loaded.
    :param saving_directory: RELATIVE path to the saving folder (must end with /).

    Return:
        - Pretrained model.
    """
    if saving_directory is None:
        saving_directory = get_cache_folder()

    if not os.path.exists(saving_directory):
        os.makedirs(saving_directory)
    
    if os.path.exists(saving_directory + "reprod/reprod.ckpt"):
        return saving_directory + "reprod/reprod.ckpt"
    
    models = {"polos" : "https://polos-polaris.s3.ap-northeast-1.amazonaws.com/reprod.zip"}

    if os.path.isdir(saving_directory + model):
        click.secho(f"{model} is already in cache.", fg="yellow")
        if not model.endswith("/"):
            model += "/"

    elif model not in models.keys():
        raise Exception(f"{model} is not a valid Polos model!")

    elif models[model].startswith("https://"):
        download_file_maybe_extract(models[model], directory=saving_directory)

    else:
        raise Exception("Something went wrong while dowloading the model!")

    if os.path.exists(saving_directory + model + ".zip"):
        os.remove(saving_directory + model + ".zip")

    click.secho("Download succeeded. Loading model...", fg="yellow")
    experiment_folder = saving_directory + "reprod"
    checkpoints = [
        file for file in os.listdir(experiment_folder) if file.endswith(".ckpt")
    ]
    checkpoint = checkpoints[-1]
    checkpoint_path = experiment_folder + "/" + checkpoint
    return checkpoint_path


def load_checkpoint(checkpoint: str) -> ModelBase:
    """Function that loads a model from a checkpoint file.
    :param checkpoint: Path to the checkpoint file.

    Returns:
        - Polos Model
    """
    if not os.path.exists(checkpoint):
        raise Exception(f"{checkpoint} file not found!")

    tags_csv_file = "/".join(checkpoint.split("/")[:-1] + ["meta_tags.csv"])
    hparam_yaml_file = "/".join(checkpoint.split("/")[:-1] + ["hparams.yaml"])

    # if os.path.exists(tags_csv_file):
    #     # Uggly convertion from older Lightning checkpoints
    #     tags = pd.read_csv(
    #         tags_csv_file, header=None, index_col=0, squeeze=True
    #     ).to_dict()
    #     hparams = {}
    #     for k, v in tags.items():
    #         if isinstance(v, str) and v.replace(".", "", 1).isdigit():
    #             hparams[k] = float(v) if "." in v else int(v)
    #         else:
    #             hparams[k] = v
    #     model = str2model[tags["model"]].load_from_checkpoint(
    #         checkpoint, hparams=hparams
    #     )
    if os.path.exists(hparam_yaml_file):

        with open(hparam_yaml_file) as yaml_file:
            hparams = yaml.load(yaml_file.read(), Loader=yaml.FullLoader)
        model_class = str2model[hparams["model"]]
        model = model_class(hparams)

        ckpt = torch.load(checkpoint)
        state_dict = ckpt.get("state_dict", ckpt)
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("model.", "")  # adjust if needed
            new_state_dict[new_key] = v

        # 5. Load weights
        model.load_state_dict(new_state_dict, strict=False)
    else:
        raise Exception(
            "[meta_tags.csv|hparams.yaml is missing from the checkpoint folder."
            " Please clean your cache folder (~/.cache/torch/yuigawada/) and try to download the model again."
        )

    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model
