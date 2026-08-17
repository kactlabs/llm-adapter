import os
import sys


def validate_env():
    """Compare .env and .env.sample keys. Show error if any mismatch."""
    sample_path = os.path.join(os.path.dirname(__file__), ".env.sample")
    env_path = os.path.join(os.path.dirname(__file__), ".env")

    if not os.path.exists(sample_path):
        print("\033[91m[ENV ERROR] .env.sample file not found!\033[0m")
        return

    if not os.path.exists(env_path):
        print("\033[91m[ENV ERROR] .env file not found! Copy .env.sample to .env and fill in values.\033[0m")
        sys.exit(1)

    sample_keys = _extract_keys(sample_path)
    env_keys = _extract_keys(env_path)

    missing_in_env = sample_keys - env_keys
    extra_in_env = env_keys - sample_keys

    if missing_in_env or extra_in_env:
        print("\033[91m[ENV MISMATCH] .env and .env.sample are out of sync:\033[0m")
        if missing_in_env:
            print(f"\033[93m  Missing in .env (present in .env.sample):\033[0m")
            for key in sorted(missing_in_env):
                print(f"    - {key}")
        if extra_in_env:
            print(f"\033[93m  Extra in .env (not in .env.sample):\033[0m")
            for key in sorted(extra_in_env):
                print(f"    - {key}")
        sys.exit(1)
    else:
        print("\033[92m[ENV OK] .env and .env.sample keys are in sync.\033[0m")


def _extract_keys(filepath):
    """Extract variable keys from an env file, ignoring comments and blank lines."""
    keys = set()
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key = line.split("=", 1)[0].strip()
                if key:
                    keys.add(key)
    return keys


if __name__ == "__main__":
    validate_env()