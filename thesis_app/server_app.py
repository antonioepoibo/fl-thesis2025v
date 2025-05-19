"""thesis-app: A Flower / PyTorch app."""

from flwr.common import Context, ndarrays_to_parameters, Metrics
from typing import List, Tuple
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from thesis_app.my_strategy import CustomFedAvg
from thesis_app.FedAvgEnc import FedAvgEnc
from thesis_app.task import Net, get_weights

def average_metric(metrics: List[Tuple[int, Metrics]], key: str) -> float:
    """Compute the average of a specific metric key."""
    values = [m[key] for _, m in metrics if key in m]
    if not values:
        return 0.0
    return sum(values) / len(values)

def aggregate_fit_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate fit metrics from all clients."""
    print("[Server] Aggregating client fit metrics...")

    if not metrics:
        return {}

    avg_train_time = average_metric(metrics, "train_time")

    return {
        "avg_train_time": avg_train_time
    }

def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize model parameters
    weights = get_weights(Net())
    shapes = [w.shape for w in weights]
    parameters = ndarrays_to_parameters(weights)

    # Define strategy
    strategy = FedAvgEnc(
        shapes=shapes,
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
