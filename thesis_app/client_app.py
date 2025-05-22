import torch
import time
import psutil
from flwr.client import ClientApp, NumPyClient,Client
from flwr.common import Context
from thesis_app.task import Net, get_weights,get_weights_shapes, load_data, set_weights, test, train
from thesis_app.comunication_utils import weights_to_parameters, serialized_to_weights
from flwr.common import (
    Parameters,
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)


class FlowerClient(Client):
    def __init__(self, net, trainloader, valloader, local_epochs):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        print("[Client] get_parameters CALLED")

        weights = get_weights(self.net)

        parameters=weights_to_parameters(weights)

        status = Status(code=Code.OK, message="Success")
        return GetParametersRes(
            status=status,
            parameters=parameters
        )

    def fit(self, ins: FitIns) -> FitRes:
        print("[Client] fit CALLED")
        # Step 1: Deserialize parameters and update local model
        shapes = get_weights_shapes(self.net)
        weights=serialized_to_weights(ins.parameters,shapes)
        set_weights(self.net, weights)

        # Step 2: Initialize monitoring tools
        process = psutil.Process()
        process.cpu_percent(None)  # Prime CPU counter

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Step 3: Train the model locally
        start_time = time.time()
        train(self.net, self.trainloader, epochs=self.local_epochs, device=self.device)
        train_time = time.time() - start_time

        # Step 4: Collect system metrics
        cpu_percent = process.cpu_percent(interval=None)  # % CPU used since last call
        memory_usage = process.memory_info().rss / 1024 / 1024  # in MB
        normalized_cpu = cpu_percent / psutil.cpu_count()


        if torch.cuda.is_available():
            gpu_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # in MB
        else:
            gpu_memory = 0.0

        # Step 5: Get updated model weights
        ndarrays_updated = get_weights(self.net)
        #print(f"[Client] Updated model weights (first 10 values of first tensor): {ndarrays_updated[0].flatten()[:10]}")

        # Step 6: Encrypt and serialize updated model weights
        parameters_updated = weights_to_parameters(ndarrays_updated)

        metrics = {
            "train_time": train_time,
            "memory_usage_mb": memory_usage,
            "cpu_percent": normalized_cpu,
            "gpu_memory_mb": gpu_memory,
        }

        print(f"[Client] Fit metrics: {metrics}")

        return FitRes(
            status=Status(code=Code.OK, message="Encrypted model update"),
            parameters=parameters_updated,
            num_examples=len(self.trainloader),
            metrics=metrics,
        )



    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        print("[Client] evaluate CALLED")
        # Deserialize parameters to NumPy ndarray's
        parameters_original = ins.parameters
        ndarrays_original = parameters_to_ndarrays(parameters_original)

        set_weights(self.net, ndarrays_original)
        loss, accuracy = test(self.net, self.valloader, self.device)

        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return EvaluateRes(
            status=status,
            loss=float(loss),
            num_examples=len(self.valloader),
            metrics={"accuracy": float(accuracy)},
        )



def client_fn(context: Context):
    # Load model and data
    net = Net()
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions)
    local_epochs = context.run_config["local-epochs"]

    # Return Client instance
    return FlowerClient(net, trainloader, valloader, local_epochs).to_client()


# Create the ClientApp
app = ClientApp(client_fn=client_fn)
