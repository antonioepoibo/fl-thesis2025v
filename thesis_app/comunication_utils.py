import numpy as np
from thesis_app.create_enc_context_CKKS import (
    encrypt_tensors,
    decrypt_tensors,
    serialize_encrypted_tensors,
    deserialize_encrypted_tensors,
    load_context,
)
from flwr.common import Parameters

enc_context=load_context("./context_data/ckks.context")

def serialized_to_weights(parameters: Parameters,shapes):
    """"""
    if parameters.tensor_type == "tenseal.ckks":
        deserialized = deserialize_encrypted_tensors(enc_context, parameters.tensors)
        decrypted = decrypt_tensors(deserialized)

        # Reshape to original shapes
        reshaped_weights = []
        for i, arr in enumerate(decrypted):
            reshaped = np.array(arr).reshape(shapes[i])
            reshaped_weights.append(reshaped)

        return reshaped_weights
        
def encrypted_to_parameters(encrypted):
    """"""
    serialized = serialize_encrypted_tensors(encrypted)

    # Step 4: Package into FL Parameters
    parameters = Parameters(
        tensors=serialized,
        tensor_type="tenseal.ckks"
    )

    return parameters

def get_encrypted(tensors):
    return deserialize_encrypted_tensors(enc_context, tensors)

def weights_to_parameters(weights):
    """"""
    flattened_ndarrays = [arr.flatten() for arr in weights]  # Flatten each tensor
    encrypted = encrypt_tensors(enc_context, flattened_ndarrays)
    parameters=encrypted_to_parameters(encrypted)
    return parameters
