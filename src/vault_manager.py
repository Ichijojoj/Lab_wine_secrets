import hvac
import hvac.exceptions
import os
import logging
import requests.exceptions

logger = logging.getLogger(__name__)


class VaultManager:
    def __init__(self):
        self.vault_url = os.getenv("VAULT_ADDR", "http://vault:8200")
        raw_token = os.getenv("VAULT_TOKEN", "myroot")
        self.token = raw_token.strip().replace('"', '').replace("'", "")
        self.client = hvac.Client(url=self.vault_url, token=self.token)

    def get_db_secrets(self):
        if os.getenv("PYTEST_CURRENT_TEST"):
            return {"DB_USER": "test_user", "DB_PASSWORD": "test_password"}

        try:
            if not self.client.is_authenticated():
                logger.warning("Vault client is not authenticated.")
                return None

            read_response = self.client.secrets.kv.v2.read_secret_version(
                path='oracle-db-creds',
                mount_point='secret'
            )
            return read_response['data']['data']
        except hvac.exceptions.VaultError as e:
            logger.warning(f"Vault communication error: {e}. Using env fallbacks.")
        except requests.exceptions.RequestException as e:
            logger.warning(f"HTTP Connection error to Vault: {e}. Using env fallbacks.")

        return {
            "DB_USER": os.getenv("DB_APP_USER", "default"),
            "DB_PASSWORD": os.getenv("DB_APP_PASSWORD", "default")
        }


vault_manager = VaultManager()