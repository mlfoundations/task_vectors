import argparse
from itertools import combinations

import torch
from ray import air, tune
from ray.air import session
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.bayesopt import BayesOptSearch

from src.eval import eval_single_dataset
from src.task_vectors import (
    TaskVector,
    TaskVectorTopKZero,
    TaskVectorTopKInit,
    TaskVectorTopKKeep,
    TaskVectorMiddleKeep,
)

zeroshot_acc = {
    "ViT-B-32": {
        "MNIST": 48.25,
        "RESISC45": 60.22,
        "DTD": 44.41,
        "GTSRB": 32.56,
        "SVHN": 31.61,
        "SUN397": 62.92,
        "EuroSAT": 45.15,
        "Cars": 59.64,
    },
    "ViT-B-16": {
        "MNIST": 51.80,
        "RESISC45": 66.35,
        "DTD": 44.68,
        "GTSRB": 43.34,
        "SVHN": 51.98,
        "SUN397": 65.22,
        "EuroSAT": 54.52,
        "Cars": 64.71,
    },
    "ViT-L-14": {
        "MNIST": 76.36,
        "RESISC45": 71.33,
        "DTD": 55.37,
        "GTSRB": 50.55,
        "SVHN": 58.45,
        "SUN397": 67.96,
        "EuroSAT": 61.63,
        "Cars": 77.94,
    },
}
finetuned_acc = {
    "ViT-B-32": {
        "MNIST": 99.69,
        "RESISC45": 96.11,
        "DTD": 79.41,
        "GTSRB": 98.73,
        "SVHN": 97.46,
        "SUN397": 74.98,
        "EuroSAT": 99.70,
        "Cars": 77.66,
    },
    "ViT-B-16": {
        "MNIST": 99.76,
        "RESISC45": 96.89,
        "DTD": 82.07,
        "GTSRB": 99.17,
        "SVHN": 97.86,
        "SUN397": 78.20,
        "EuroSAT": 99.70,
        "Cars": 86.79,
    },
    "ViT-L-14": {
        "MNIST": 99.69,
        "RESISC45": 97.37,
        "DTD": 84.15,
        "GTSRB": 99.24,
        "SVHN": 98.11,
        "SUN397": 81.96,
        "EuroSAT": 99.85,
        "Cars": 92.39,
    },
}


def evaluate_all_datasets(args: argparse.Namespace, image_encoder: torch.nn.Module) -> float:
    average_normalized_acc = 0.0
    for dataset in args.data_sets:
        results = eval_single_dataset(image_encoder, dataset, args)["top1"] * 100.0
        normalized_acc = (results / finetuned_acc[args.model][dataset]) * 100.0
        average_normalized_acc += normalized_acc
    return average_normalized_acc / len(args.data_sets)


def evaluate_on_task_subsets(config, args: argparse.Namespace):
    if args.run_name == "paper_implementation":
        task_vectors_dict = {
            dataset: TaskVector(
                args.pretrained_checkpoint, f"{args.checkpoint_path}/{args.model}/{dataset}/finetuned.pt"
            )
            for dataset in args.data_sets
        }
    elif args.run_name == "topk_zero":
        task_vectors_dict = {
            dataset: TaskVectorTopKZero(
                pretrained_checkpoint=args.pretrained_checkpoint,
                finetuned_checkpoint=f"{args.checkpoint_path}/{args.model}/{dataset}/finetuned.pt",
                top_k=config["beta"],
            )
            for dataset in args.data_sets
        }
    elif args.run_name == "topk_init":
        task_vectors_dict = {
            dataset: TaskVectorTopKInit(
                pretrained_checkpoint=args.pretrained_checkpoint,
                finetuned_checkpoint=f"{args.checkpoint_path}/{args.model}/{dataset}/finetuned.pt",
                top_k=config["beta"],
            )
            for dataset in args.data_sets
        }
    elif args.run_name == "topk_keep":
        task_vectors_dict = {
            dataset: TaskVectorTopKKeep(
                pretrained_checkpoint=args.pretrained_checkpoint,
                finetuned_checkpoint=f"{args.checkpoint_path}/{args.model}/{dataset}/finetuned.pt",
                top_k=config["beta"],
            )
            for dataset in args.data_sets
        }
    elif args.run_name == "middle_keep":
        task_vectors_dict = {
            dataset: TaskVectorMiddleKeep(
                pretrained_checkpoint=args.pretrained_checkpoint,
                finetuned_checkpoint=f"{args.checkpoint_path}/{args.model}/{dataset}/finetuned.pt",
                top_k_keep=config["beta"],
                top_k_remove=config["gamma"],
            )
            for dataset in args.data_sets
        }
    else:
        raise ValueError("Unsupported method of task vectors.")

    global_normalized_acc = 0.0
    for index, data_subsets in enumerate(combinations(args.data_sets, args.evaluation_depth)):
        task_vectors = [task_vectors_dict[dataset] for dataset in data_subsets]
        task_vector_sum = sum(task_vectors)
        image_encoder = task_vector_sum.apply_to(args.pretrained_checkpoint, scaling_coef=config["alpha"])

        avg_normalized_acc = evaluate_all_datasets(args=args, image_encoder=image_encoder)
        if index == 0:
            global_normalized_acc = avg_normalized_acc
        else:
            global_normalized_acc = global_normalized_acc * (index / (index + 1)) + avg_normalized_acc / (index + 1)

        session.report({"global_normalized_acc": global_normalized_acc})


def main(args: argparse.Namespace):
    # build and load all the needed task vectors at once
    if args.run_name == "paper_implementation":
        space = {"alpha": tune.choice(list(x / 10.0 for x in range(1, 11)))}
        num_samples = 10
    elif args.run_name == "topk_zero":
        space = {
            "alpha": tune.choice(list(x / 10.0 for x in range(1, 11))),
            "beta": tune.choice([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]),
        }
        num_samples = 40
    elif args.run_name == "topk_init":
        space = {
            "alpha": tune.choice(list(x / 10.0 for x in range(1, 11))),
            "beta": tune.choice([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]),
        }
        num_samples = 40
    elif args.run_name == "topk_keep":
        space = {
            "alpha": tune.choice(list(x / 10.0 for x in range(1, 11))),
            "beta": tune.choice([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]),
        }
        num_samples = 40
    elif args.run_name == "middle_keep":
        space = {
            "alpha": tune.choice(list(x / 10.0 for x in range(1, 11))),
            "beta": tune.choice([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]),
            "gamma": tune.choice(list(x / 1000.0 for x in range(1, 11))),
        }
        num_samples = 100
    else:
        raise ValueError("Unsupported method of task vectors.")

    asha_scheduler = ASHAScheduler(
        time_attr="training_iteration",
        metric="global_normalized_acc",
        mode="max",
        max_t=10,
        grace_period=3,
        reduction_factor=4,
        brackets=1,
    )
    algo = BayesOptSearch(metric="global_normalized_acc", mode="max", random_search_steps=10)
    tuner = tune.Tuner(
        tune.with_parameters(evaluate_on_task_subsets, args=args),
        param_space=space,
        tune_config=tune.TuneConfig(num_samples=num_samples, scheduler=asha_scheduler, search_alg=algo),
        run_config=air.RunConfig(
            callbacks=[
                WandbLoggerCallback(project=f"parameter_search_{args.model}_{args.method}_depth{args.evaluation_depth}")
            ]
        ),
    )
    tuner.fit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_location",
        help="Path to the root data.",
        default="~/data",
        type=str,
    )
    parser.add_argument(
        "--method",
        help="Optional name for the run.",
        type=str,
        default="paper_implementation",
        choices=["paper_implementation", "topk_zero", "topk_init", "topk_keep", "middle_keep"],
    )
    parser.add_argument(
        "--checkpoint_path",
        help="Path to the directory that holds all the checkpoints.",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--model",
        help="The type of model.",
        choices=["ViT-B-32", "ViT-B-16", "ViT-L-14"],
        default="ViT-B-32",
        type=str,
    )
    parser.add_argument(
        "--data_sets",
        help="The name of the datasets used for evaluation",
        choices=["MNIST", "Cars", "RESISC45", "DTD", "GTSRB", "SVHN", "EuroSAT", "SUN397"],
        default="MNIST",
        nargs="*",
        type=str,
    )

    parser.add_argument(
        "--evaluation_depth",
        help="The depth refers to how many task vectors should be added for evaluation.",
        default=1,
        type=int,
    )

    parser.add_argument(
        "--alpha",
        help="The value of alpha indicates the task vector multipliers.",
        default=0.4,
        type=float,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--beta",
        help="The removal value.",
        default=0.0,
        type=float,
    )
    parser.add_argument(
        "--gamma",
        help="The removal value needed for the middle keep method.",
        default=0.0,
        type=float,
    )

    parser.add_argument(
        "--eval_on_partial_datasets",
        help="If used, we evaluate only on the datasets relevant to the task vectors as opposed to all the datasets.",
        default=False,
        action="store_true",
    )

    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    args.save = f"{args.checkpoint_path}/{args.model}"
    args.pretrained_checkpoint = f"{args.checkpoint_path}/{args.model}/zeroshot.pt"
    main(args=args)