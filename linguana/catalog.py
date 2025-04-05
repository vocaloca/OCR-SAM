from google.auth.transport.requests import Request
from google.oauth2 import id_token
import google.auth.transport.requests
import requests
import os


BASE_URL = "https://cms.linguana.com/transcoder"
CLIENT_ID = "357470755328-pfjh75a2hldmng2u1rjkepqn5a5cucs4.apps.googleusercontent.com"
# 357470755328-pfjh75a2hldmng2u1rjkepqn5a5cucs4.apps.googleusercontent.com


# send Patch request to catalog
def send_patch_request(videoId, payload):
    url = f"{BASE_URL}/videos/{videoId}"
    credentials = fixed_fetch_id_token_credentials(CLIENT_ID)
    headers = {
        "Authorization": f"Bearer {credentials}",
        "Content-Type": "application/json",
    }
    response = requests.request("PATCH", url, headers=headers, data=payload)
    response.raise_for_status()
    print(f"Patch request sent to {url}")


def fixed_fetch_id_token_credentials(audience: str, request=None):
    """Get OpenID credentials from the current environment.

    NOTE: This is needed only because google.oauth2.id_token.fetch_id_token_credentials doesn't support impersonated
    credentials. Once it does, this function can be removed.
    """
    if not os.getenv("LOCAL_DEV"):
        return id_token.fetch_id_token(Request(), CLIENT_ID)
    creds, _project_id = google.auth.default()

    if request is None:
        request = google.auth.transport.requests.Request()

    if isinstance(creds, google.auth.impersonated_credentials.Credentials):
        id_creds = google.auth.impersonated_credentials.IDTokenCredentials(
            creds, audience, include_email=True
        )

    elif isinstance(creds, google.oauth2.service_account.Credentials):
        id_creds = google.oauth2.service_account.IDTokenCredentials(
            signer=creds.signer,
            service_account_email=creds.service_account_email,
            token_uri=creds._token_uri,
            quota_project_id=creds.quota_project_id,
            target_audience=audience,
        )

    elif isinstance(creds, google.auth.compute_engine.credentials.Credentials):
        id_creds = google.auth.compute_engine.credentials.IDTokenCredentials(
            request,
            audience,
            use_metadata_identity_endpoint=True,
            quota_project_id=creds.quota_project_id,
        )

    elif isinstance(creds, google.oauth2.credentials.Credentials):
        raise ValueError(
            "IDTokens are not supported for human accounts. Provide a service account instead."
        )

    else:
        raise ValueError(f"Unknown credentials type {type(creds)}")
    id_creds.refresh(Request())
    return id_creds.token
