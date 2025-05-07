import hashlib
import pickle
import os

def get_hash(params):
    """Create a hash from the simulation parameters."""
    param_str = str(params)
    return hashlib.md5(param_str.encode()).hexdigest()

def save_result(result, hash_value, prefix="result"):
    """Save the result to a file."""
    filename = f"cache/{prefix}_{hash_value}.pkl"
    os.makedirs("cache", exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(result, f)

def load_result(hash_value, prefix="result"):
    """Load the result from a file if it exists."""
    filename = f"cache/{prefix}_{hash_value}.pkl"
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return None

def clear_cache(prefix=None):
    """Clear all cached results or those with a specific prefix."""
    if not os.path.exists("cache"):
        return
    
    if prefix is None:
        # Clear all cache files
        for file in os.listdir("cache"):
            os.remove(os.path.join("cache", file))
    else:
        # Clear only files with specific prefix
        for file in os.listdir("cache"):
            if file.startswith(f"{prefix}_"):
                os.remove(os.path.join("cache", file)) 