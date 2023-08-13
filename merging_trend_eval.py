import wandb
import argparse
from tqdm.auto import tqdm
from itertools import combinations
from src.task_vectors import TaskVector, TaskVectorTopKZero
from src.eval import eval_single_dataset
import torch

zeroshot_acc = {"MNIST": 48.25, "RESISC45": 60.22, "DTD": 44.41, "GTSRB": 32.56, "SVHN": 31.61}
finetuned_acc = {"MNIST": 99.69, "RESISC45": 96.11, "DTD": 79.41, "GTSRB": 98.73, "SVHN": 97.46}


def main(args: argparse.Namespace):
    wandb.init(
        # Set the project where this run will be logged
        project="task-vector-addition",
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=f"{args.run_name}_alpha_{args.alpha}_beta_{args.beta}",
        # Track hyperparameters and run metadata
        config={"model": "ViT-B-32", "alpha": args.alpha, "method": args.run_name, "beta": args.beta},
    )
    for nb_datasets in tqdm(range(args.evaluation_depth + 1)):
        global_average_acc = 0.0
        global_average_normalized_acc = 0.0
        nb_subset = 0
        for data_subsets in tqdm(combinations(args.data_sets, nb_datasets)):
            alpha = args.alpha
            average_acc = 0.0
            average_normalized_acc = 0.0

            if len(data_subsets) == 0:
                data_subsets = args.data_sets
                alpha = 0
            if args.run_name == "paper_implementation":
                task_vectors = [
                    TaskVector(
                        args.pretrained_checkpoint, f"{args.checkpoint_path}/{args.model}/{dataset}/finetuned.pt"
                    )
                    for dataset in data_subsets
                ]
            elif args.run_name == "topk_zero":
                task_vectors = [
                    TaskVectorTopKZero(
                        pretrained_checkpoint=args.pretrained_checkpoint,
                        finetuned_checkpoint=f"{args.checkpoint_path}/{args.model}/{dataset}/finetuned.pt",
                        top_k=args.beta,
                    )
                    for dataset in data_subsets
                ]
            else:
                raise ValueError("Unsupported method of task vectors.")
            task_vector_sum = sum(task_vectors)
            image_encoder = task_vector_sum.apply_to(args.pretrained_checkpoint, scaling_coef=alpha)
            for dataset in data_subsets:
                results = eval_single_dataset(image_encoder, dataset, args)["top1"] * 100.0
                average_normalized_acc += (results / finetuned_acc[dataset]) * 100.0
                average_acc += results

            average_acc /= len(data_subsets)
            average_normalized_acc /= len(data_subsets)
            global_average_acc += average_acc
            global_average_normalized_acc += average_normalized_acc
            wandb.log(
                {
                    "average_acc": average_acc,
                    "average_normalized_acc": average_normalized_acc,
                    "nb_task_vectors": nb_datasets,
                    "tasks": " ".join([str(t) for t in data_subsets]),
                }
            )
            nb_subset += 1
        global_average_acc /= nb_subset
        global_average_normalized_acc /= nb_subset
        wandb.log(
            {
                "global_average_acc": global_average_acc,
                "global_average_normalized_acc": global_average_normalized_acc,
                "nb_task_vectors": nb_datasets,
            }
        )
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_location",
        help="Path to the root data.",
        default="~/data",
        type=str,
    )
    parser.add_argument(
        "--run_name",
        help="Optional name for the run.",
        type=str,
        default="paper_implementation",
        choices=["paper_implementation", "topk_zero"],
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
        choices=["ViT-B-32", "ViT-B-16", "ViT-L-16"],
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
        default=0,
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
        default=128,
    )

    parser.add_argument(
        "--beta",
        help="The removal value.",
        default=0.0,
        type=float,
    )

    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    args.save = f"{args.checkpoint_path}/{args.model}"
    args.pretrained_checkpoint = f"{args.checkpoint_path}/{args.model}/zeroshot.pt"
    main(args=args)
