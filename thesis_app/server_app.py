"""thesis-app: A Flower / PyTorch app."""

from flwr.common import Context, ndarrays_to_parameters, Metrics, Parameters
from typing import List, Tuple
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from thesis_app.my_strategy import CustomFedAvg
from thesis_app.FedAvgEnc import FedAvgEnc
from thesis_app.task import Net, set_weights, test, get_transforms

from typing import List, Tuple
from flwr.common import Metrics
from datasets import load_dataset
from torch.utils.data import DataLoader

def get_evaluate_fn(testloader, device):
    
    def evaluate(server_round, parameters_ndarrays, config):
        """Return a callback function for server-side evaluation."""
        # Load model
        net = Net()
        set_weights(net, parameters_ndarrays)
        net.to(device)

        loss, accuracy = test(net, testloader, device=device)

        # Return evaluation metrics
        return loss,{"cen_accuracy": accuracy,        }


    return evaluate

def average_metric(metrics: List[Tuple[int, Metrics]], key: str) -> float:
    """Compute the average of a specific metric key."""
    values = [m[key] for _, m in metrics if key in m]
    if not values:
        return 0.0
    return sum(values) / len(values)


def fit_metrics_aggregation_fn(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate fit metrics from all clients."""
    print("[Server] Aggregating client fit metrics...")

    if not metrics:
        return {}

    return {
        "avg_train_time": average_metric(metrics, "train_time"),
        "avg_memory_usage_mb": average_metric(metrics, "memory_usage_mb"),
        "avg_cpu_percent": average_metric(metrics, "cpu_percent"),
        "avg_gpu_memory_mb": average_metric(metrics, "gpu_memory_mb"),
    }


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    
    # load global test set
    testset = load_dataset("uoft-cs/cifar10")["test"]
    testloader= DataLoader(testset.with_transform(get_transforms()), batch_size=32, shuffle=False)
    

    # Define strategy
    strategy = FedAvgEnc(
        fraction_fit=fraction_fit,
        fraction_evaluate=0.0, #Note: evaluation needs to be updated to use client side 
        min_available_clients=2,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_fn=get_evaluate_fn(testloader, device="cpu"),  
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
