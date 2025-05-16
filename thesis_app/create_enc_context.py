import tenseal as ts
import os

# -------------------------------
# Context Handling
# -------------------------------
def create_ckks_context():
    poly_mod_degree = 8192
    coeff_mod_bit_sizes = [60, 40, 40, 60]
    global_scale = 2 ** 40

    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_mod_degree,
        coeff_mod_bit_sizes=coeff_mod_bit_sizes
    )
    context.global_scale = global_scale
    context.generate_galois_keys()
    context.generate_relin_keys()
    return context

def save_context(context, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        f.write(context.serialize(
            save_public_key=True,
            save_secret_key=True,
            save_galois_keys=True,
            save_relin_keys=True
        ))

def load_context(filepath):
    with open(filepath, "rb") as f:
        return ts.context_from(f.read())

# -------------------------------
# Encryption / Decryption
# -------------------------------
def encrypt_tensors(context, tensors):
    return [ts.ckks_vector(context, tensor) for tensor in tensors]

def decrypt_tensors(encrypted_vectors):
    return [vec.decrypt() for vec in encrypted_vectors]

# -------------------------------
# Serialization / Deserialization
# -------------------------------
def serialize_encrypted_tensors(encrypted_vectors):
    return [vec.serialize() for vec in encrypted_vectors]

def deserialize_encrypted_tensors(context, serialized_list):
    return [ts.ckks_vector_from(context, data) for data in serialized_list]

# -------------------------------
# Example Usage (All In-Memory)
# -------------------------------
if __name__ == "__main__":
    # Example tensors (list of float lists)
    tensors = [
        [1.1, 2.2, 3.3],
        [4.4, 5.5, 6.6],
        [7.7, 8.8, 9.9]
    ]

    # Create and save context
    context = create_ckks_context()
    save_context(context, "./context_data/ckks.context")

    # Load context from disk
    loaded_context = load_context("./context_data/ckks.context")

    # Encrypt using loaded context
    encrypted = encrypt_tensors(loaded_context, tensors)

    # Serialize in memory
    serialized = serialize_encrypted_tensors(encrypted)

    # Deserialize in memory
    deserialized = deserialize_encrypted_tensors(loaded_context, serialized)

    # Decrypt and verify
    decrypted = decrypt_tensors(deserialized)
    for i, tensor in enumerate(decrypted):
        print(f"Decrypted tensor {i + 1}: {tensor}")
