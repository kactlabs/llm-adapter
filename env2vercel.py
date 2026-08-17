import subprocess
import sys


def parse_env(filepath=".env"):
    """Parse .env file, skipping comments and blank lines."""
    env_vars = {}
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            value = value.strip()
            # Strip surrounding quotes (single or double)
            if (value.startswith('"') and value.endswith('"')) or \
               (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]
            env_vars[key.strip()] = value
    return env_vars


def get_existing_env_vars(environment="production"):
    """Get list of env var names already set in Vercel."""
    result = subprocess.run(
        ["vercel", "env", "ls", environment],
        capture_output=True, text=True
    )
    return result.stdout


def sync_env_to_vercel(filepath=".env", environment="production"):
    """Sync all variables from .env to Vercel, removing existing ones first."""
    env_vars = parse_env(filepath)
    existing_output = get_existing_env_vars(environment)

    if not env_vars:
        print("No variables found in", filepath)
        return

    print(f"Found {len(env_vars)} variable(s) to sync to '{environment}':\n")

    for key, value in env_vars.items():
        already_exists = key in existing_output

        if already_exists:
            print(f"  [{key}] exists — removing first...")
            subprocess.run(
                ["vercel", "env", "rm", key, environment, "-y"],
                capture_output=True, text=True
            )

        print(f"  [{key}] adding...")
        result = subprocess.run(
            ["vercel", "env", "add", key, environment],
            input=value, capture_output=True, text=True
        )

        if result.returncode == 0:
            print(f"  [{key}] ✓ done")
        else:
            print(f"  [{key}] ✗ failed: {result.stderr.strip()}")
            print("\nAborting — fix the issue and retry.")
            sys.exit(1)

    print("\nSync complete.")


if __name__ == "__main__":
    env_file = sys.argv[1] if len(sys.argv) > 1 else ".env"
    env_target = sys.argv[2] if len(sys.argv) > 2 else "production"
    sync_env_to_vercel(env_file, env_target)