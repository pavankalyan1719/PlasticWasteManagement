import getpass
import hashlib
import secrets
from pathlib import Path

config_dir = Path(".streamlit")
config_dir.mkdir(exist_ok=True)

secrets_file = config_dir / "secrets.toml"

username = input("Enter admin username: ").strip()

if not username:
    raise SystemExit("Username cannot be empty.")

password = getpass.getpass("Create admin password: ")

if len(password) < 10:
    raise SystemExit("Use a password with at least 10 characters.")

confirmation = getpass.getpass("Confirm admin password: ")

if password != confirmation:
    raise SystemExit("Passwords do not match.")

salt = secrets.token_hex(32)

password_hash = hashlib.pbkdf2_hmac(
    "sha256",
    password.encode("utf-8"),
    salt.encode("utf-8"),
    200000
).hex()

# Preserve existing settings such as GEOAPIFY_API_KEY.
existing_lines = []

if secrets_file.exists():
    existing_lines = [
        line
        for line in secrets_file.read_text(
            encoding="utf-8"
        ).splitlines()
        if not line.strip().startswith(
            (
                "ADMIN_USERNAME",
                "ADMIN_SALT",
                "ADMIN_PASSWORD_HASH"
            )
        )
    ]

admin_lines = [
    f'ADMIN_USERNAME = "{username}"',
    f'ADMIN_SALT = "{salt}"',
    f'ADMIN_PASSWORD_HASH = "{password_hash}"'
]

secrets_file.write_text(
    "\n".join(existing_lines + admin_lines) + "\n",
    encoding="utf-8"
)

print("Admin credentials configured successfully.")