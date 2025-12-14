## SSH tunnel for Vast.ai

Set these values (defaults shown):

```bash
VAST_SSH_USER=root
VAST_SSH_HOST=<ip>
VAST_SSH_PORT=<vast-ssh-port>
VAST_API_PORT=9880
LOCAL_FORWARD_PORT=18081
SSH_KEY_PATH=/root/.ssh/vast_key
```

### Vast: run the API

```bash
cd /workspace/GPT-SoVITS
python3 api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml
```

### Backend (Ubuntu): install tools

```bash
sudo apt-get update
sudo apt-get install -y autossh
```

### Backend (Ubuntu): generate an SSH key (recommended)

```bash
sudo mkdir -p /root/.ssh
sudo ssh-keygen -t ed25519 -f /root/.ssh/vast_key -N ""
sudo chmod 600 /root/.ssh/vast_key
```

### Vast: authorize the backend key (terminal)

This appends your backend public key to `/root/.ssh/authorized_keys` on the Vast instance.
You must be able to SSH into the Vast instance already (using any working key). If not, add the public key via Vast UI once, or run this command from a machine that already has SSH access.
```bash
sudo ssh-copy-id -i /root/.ssh/vast_key.pub -p "$VAST_SSH_PORT" root@"$VAST_SSH_HOST"
```
Test that the new key works:

```bash
ssh -i /root/.ssh/vast_key -p "$VAST_SSH_PORT" root@"$VAST_SSH_HOST" 'echo ok'
```

### Backend (Ubuntu): start tunnel (systemd, auto-restart)

Add + start a tunnel named `vast1`:
```bash
sudo bash deploy/ssh-tunnel/gptsovits-tunnel add vast1 \
  --host "$VAST_SSH_HOST" --port "$VAST_SSH_PORT" --user "$VAST_SSH_USER" --key "$SSH_KEY_PATH" \
  --local "$LOCAL_FORWARD_PORT" --remote "$VAST_API_PORT"
```

Manage:

```bash
sudo bash deploy/ssh-tunnel/gptsovits-tunnel ls
sudo bash deploy/ssh-tunnel/gptsovits-tunnel status vast1
sudo bash deploy/ssh-tunnel/gptsovits-tunnel logs vast1
sudo bash deploy/ssh-tunnel/gptsovits-tunnel rm vast1
```

### Backend (Ubuntu): test

```bash
curl -s "http://127.0.0.1:$LOCAL_FORWARD_PORT/docs" | head
```

### Notes

- `ref_audio_path` must be a file path on the Vast machine, not on your backend.
- The tunnel stays up after you close your terminal and comes back after reboot (systemd).
