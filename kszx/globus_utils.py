"""kszx/globus_utils.py — Globus transfer helpers for kszx.

Provides a seamless download experience for datasets available via Globus
(e.g. Quijote simulations). The ``globus-sdk`` package is imported lazily,
so kszx works without it installed — a helpful error is raised if you try
to auto-download without it.

One-time user setup:
  1. Install Globus Connect Personal (GCP)
  2. Start GCP
  3. Run: globus login
  4. Run: export GLOBUS_LOCAL_ENDPOINT=$(globus endpoint local-id)
"""

import os
import time


def get_local_endpoint_id():
    """Return the user's local Globus endpoint ID.

    Checks (in order):
      1. $GLOBUS_LOCAL_ENDPOINT environment variable
      2. ``globus endpoint local-id`` CLI command (requires GCP installed)

    Raises RuntimeError with setup instructions if not configured.
    """
    ep = os.environ.get('GLOBUS_LOCAL_ENDPOINT')
    if ep:
        return ep

    # Try auto-detecting from GCP
    import subprocess
    try:
        result = subprocess.run(
            ['globus', 'endpoint', 'local-id'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            ep = result.stdout.strip()
            if ep:
                return ep
    except Exception:
        pass

    raise RuntimeError(
        'Globus local endpoint not configured.\n'
        'To enable auto-downloading of Quijote data, do the following (once):\n'
        '\n'
        '  1. Install Globus Connect Personal:\n'
        '       macOS: https://docs.globus.org/how-to/globus-connect-personal-mac/\n'
        '       Linux: https://docs.globus.org/how-to/globus-connect-personal-linux/\n'
        '  2. Start Globus Connect Personal.\n'
        '  3. Run: globus login\n'
        '  4. Run: export GLOBUS_LOCAL_ENDPOINT=$(globus endpoint local-id)\n'
        '     (Add this to your .bashrc or .zshrc to make it permanent.)\n'
    )


def get_transfer_client():
    """Return an authenticated globus_sdk.TransferClient.

    Uses tokens from 'globus login' (stored by the Globus CLI in
    ~/.globus/cli/storage.db). Uses RefreshTokenAuthorizer so that
    expired access tokens are automatically refreshed.

    Raises RuntimeError with instructions if not logged in or not installed.
    """
    try:
        import globus_sdk
    except ImportError:
        raise RuntimeError(
            'globus-sdk is not installed. To enable auto-downloading, run:\n'
            '  conda install -c conda-forge globus-sdk globus-cli\n'
        )

    token_data = _read_cli_token_data('transfer.api.globus.org')
    if token_data is None:
        raise RuntimeError(
            'Globus authentication failed — no transfer tokens found.\n'
            'Please run: globus login\n'
            'Then retry your kszx.quijote call with download=True.\n'
        )

    client_data = _read_cli_client_data()
    if client_data is None:
        raise RuntimeError(
            'Globus authentication failed — could not read CLI client data.\n'
            'Please run: globus login\n'
            'Then retry your kszx.quijote call with download=True.\n'
        )

    client = globus_sdk.NativeAppAuthClient(client_id=client_data['client_id'])
    authorizer = globus_sdk.RefreshTokenAuthorizer(
        token_data['refresh_token'],
        client,
        access_token=token_data['access_token'],
        expires_at=token_data.get('expires_at_seconds'),
    )
    return globus_sdk.TransferClient(authorizer=authorizer)


def _read_cli_token_data(resource_server):
    """Read token data for a resource server from the Globus CLI's SQLite storage.

    Returns dict with keys: access_token, refresh_token, expires_at_seconds, etc.
    Returns None if not found.
    """
    import json
    import sqlite3

    db_path = os.path.join(os.path.expanduser('~'), '.globus', 'cli', 'storage.db')
    if not os.path.exists(db_path):
        return None

    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(
            'SELECT token_data_json FROM token_storage WHERE resource_server = ?',
            (resource_server,)
        )
        row = cur.fetchone()
        conn.close()
        if row is None:
            return None
        return json.loads(row[0])
    except Exception:
        return None


def _read_cli_client_data():
    """Read the CLI's client_id from its SQLite storage.

    Returns dict with key 'client_id', or None if not found.
    """
    import json
    import sqlite3

    db_path = os.path.join(os.path.expanduser('~'), '.globus', 'cli', 'storage.db')
    if not os.path.exists(db_path):
        return None

    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(
            "SELECT config_data_json FROM config_storage WHERE config_name = 'auth_client_data'"
        )
        row = cur.fetchone()
        conn.close()
        if row is None:
            return None
        return json.loads(row[0])
    except Exception:
        return None


def globus_download(source_endpoint, remote_path, local_abspath,
                    recursive=False, label=None):
    """Download a file or directory from a Globus endpoint.

    Args:
        source_endpoint (str): Globus collection/endpoint UUID.
        remote_path (str): Path on the source endpoint.
        local_abspath (str): Absolute destination path on the local filesystem.
        recursive (bool): True for directories, False for single files.
        label (str or None): Human-readable label for the transfer task.

    This function blocks until the transfer is complete.
    Raises RuntimeError if GCP is not running or transfer fails.
    """
    import globus_sdk

    tc = get_transfer_client()
    local_ep = get_local_endpoint_id()
    gcp_path = _to_gcp_path(local_abspath)

    os.makedirs(os.path.dirname(local_abspath), exist_ok=True)

    tdata = globus_sdk.TransferData(
        source_endpoint=source_endpoint,
        destination_endpoint=local_ep,
        label=label or f'kszx: {remote_path}',
        sync_level='checksum',
    )
    tdata.add_item(remote_path, gcp_path, recursive=recursive)

    try:
        result = tc.submit_transfer(tdata)
    except globus_sdk.TransferAPIError as e:
        if 'EndpointError' in str(e):
            raise RuntimeError(
                'Globus transfer failed — is Globus Connect Personal running?\n'
                '\n'
                'To start it:\n'
                '  macOS: Open "Globus Connect Personal" from Applications.\n'
                '  Linux: ./globusconnectpersonal -start &\n'
                '\n'
                'If you haven\'t installed it yet, see:\n'
                '  https://docs.globus.org/how-to/globus-connect-personal-mac/\n'
            ) from e
        raise

    task_id = result['task_id']
    print(f'Globus transfer submitted (task_id={task_id}), waiting for completion...')
    _wait_for_task(tc, task_id)


def globus_download_batch(source_endpoint, items, label=None):
    """Download multiple files/directories in a single Globus transfer task.

    Args:
        source_endpoint (str): Globus collection/endpoint UUID.
        items (list of tuples): Each tuple is (remote_path, local_abspath, recursive).
        label (str or None): Human-readable label for the transfer task.

    This function blocks until the transfer is complete.
    """
    import globus_sdk

    tc = get_transfer_client()
    local_ep = get_local_endpoint_id()

    tdata = globus_sdk.TransferData(
        source_endpoint=source_endpoint,
        destination_endpoint=local_ep,
        label=label or 'kszx bulk download',
        sync_level='checksum',
    )

    for remote_path, local_abspath, recursive in items:
        os.makedirs(os.path.dirname(local_abspath), exist_ok=True)
        gcp_path = _to_gcp_path(local_abspath)
        tdata.add_item(remote_path, gcp_path, recursive=recursive)

    result = tc.submit_transfer(tdata)
    task_id = result['task_id']
    print(f'Globus transfer submitted ({len(items)} items, task_id={task_id}), '
          f'waiting for completion...')
    _wait_for_task(tc, task_id)


def _wait_for_task(tc, task_id, poll_interval=2.0, timeout=3600, fault_threshold=3):
    """Poll a Globus task until it completes or fails.

    Detects persistent errors (e.g. FILE_NOT_FOUND) that Globus reports as
    ACTIVE-with-faults rather than FAILED. If ``is_ok`` is False for
    ``fault_threshold`` consecutive polls, the task is cancelled and an error
    is raised.
    """
    elapsed = 0.0
    consecutive_not_ok = 0
    while True:
        task = tc.get_task(task_id)
        status = task['status']
        if status == 'SUCCEEDED':
            print(f'Globus transfer complete.')
            return
        if status == 'FAILED':
            raise RuntimeError(
                f'Globus transfer failed (task_id={task_id}).\n'
                f'Nice error message: {task.get("nice_status_short_description", "unknown")}\n'
                f'Check details at: https://app.globus.org/activity/{task_id}\n'
            )
        # Detect persistent faults (e.g. FILE_NOT_FOUND) that stay ACTIVE
        if not task.get('is_ok', True):
            consecutive_not_ok += 1
            if consecutive_not_ok >= fault_threshold:
                nice = task.get('nice_status', 'unknown')
                nice_desc = task.get('nice_status_short_description', 'unknown')
                try:
                    tc.cancel_task(task_id)
                except Exception:
                    pass
                raise RuntimeError(
                    f'Globus transfer error: {nice} ({nice_desc})\n'
                    f'task_id={task_id}\n'
                    f'Check details at: https://app.globus.org/activity/{task_id}\n'
                )
        else:
            consecutive_not_ok = 0
        time.sleep(poll_interval)
        elapsed += poll_interval
        if elapsed >= timeout:
            raise RuntimeError(
                f'Globus transfer timed out after {timeout}s (task_id={task_id}).\n'
                f'The transfer may still be running. Check: https://app.globus.org/activity/{task_id}\n'
            )


def _to_gcp_path(local_abspath):
    """Convert an absolute local path to a GCP-relative path.

    GCP uses '/~/' as a prefix for the user's home directory.
    """
    home = os.path.expanduser('~')
    if local_abspath.startswith(home):
        return '/~/' + os.path.relpath(local_abspath, home)
    return local_abspath
