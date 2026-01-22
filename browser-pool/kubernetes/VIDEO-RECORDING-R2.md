# Video Recording with Cloudflare R2 Storage

This guide explains how to enable video recording in Selenium Grid with automatic upload to Cloudflare R2.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        SELENIUM GRID POD                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐       │
│  │  Browser Node    │  │  Video Recorder  │  │   R2 Uploader    │       │
│  │  (Chrome/FF/Edge)│  │  (ffmpeg sidecar)│  │  (aws-cli sidecar)│       │
│  │                  │  │                  │  │                  │       │
│  │  Runs tests      │──│  Records screen  │──│  Uploads to R2   │       │
│  │                  │  │  to /videos/     │  │  on completion   │       │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘       │
│           │                     │                     │                  │
│           └─────────────────────┴─────────────────────┘                  │
│                         Shared /videos volume                            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ S3-compatible upload
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         CLOUDFLARE R2                                    │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  argus-artifacts bucket                                          │    │
│  │  /videos/video_abc123_20260122_123456.webm                      │    │
│  │  /videos/video_def456_20260122_123457.webm                      │    │
│  │  ...                                                             │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ Served via Cloudflare Worker
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     CLOUDFLARE WORKER                                    │
│  GET /videos/{artifact_id}?sig=SIGNATURE&exp=EXPIRATION                 │
│  - Verifies HMAC signature                                              │
│  - Returns video with proper headers                                    │
│  - Zero egress fees!                                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

## Prerequisites

1. **Cloudflare R2 Bucket** - Create at https://dash.cloudflare.com/?to=/:account/r2
2. **R2 API Token** - Create with "Edit" permissions for your bucket
3. **Kubernetes Cluster** - Vultr VKE or similar
4. **Helm** - For Selenium Grid deployment

## Quick Start

### Step 1: Create R2 Credentials

1. Go to Cloudflare Dashboard > R2 > Manage R2 API Tokens
2. Create a new token with:
   - Permission: Object Read & Write
   - Scope: Specific bucket → `argus-artifacts`
3. Copy the Access Key ID and Secret Access Key

### Step 2: Configure Kubernetes Secrets

Edit `r2-video-config.yaml` with your credentials:

```bash
# Edit the file
vim kubernetes/r2-video-config.yaml

# Replace:
# - YOUR_CLOUDFLARE_ACCOUNT_ID
# - YOUR_R2_ACCESS_KEY_ID
# - YOUR_R2_SECRET_ACCESS_KEY

# Apply to cluster
kubectl apply -f kubernetes/r2-video-config.yaml -n selenium-grid
```

### Step 3: Deploy Selenium Grid with Video Recording

```bash
# Add Selenium Helm repo
helm repo add selenium https://www.selenium.dev/docker-selenium
helm repo update

# Deploy with video recording enabled
helm upgrade --install selenium-grid selenium/selenium-grid \
  -f selenium-grid/values-with-video.yaml \
  -n selenium-grid \
  --create-namespace
```

### Step 4: Verify Deployment

```bash
# Check pods (each node should have 3 containers: browser, video-recorder, r2-uploader)
kubectl get pods -n selenium-grid

# Check logs for R2 uploader
kubectl logs -n selenium-grid selenium-chrome-node-xxx -c r2-uploader

# Test video recording
curl -X POST http://<GRID_URL>/wd/hub/session \
  -H "Content-Type: application/json" \
  -d '{"capabilities":{"browserName":"chrome","se:recordVideo":true}}'
```

## Configuration Options

### Video Quality Settings

In `values-with-video.yaml`:

```yaml
chromeNode:
  videoRecording:
    enabled: true
    screenSize: "1920x1080"  # Resolution (1920x1080, 1280x720, etc.)
    frameRate: 15            # FPS (15-30 recommended)
```

### R2 Upload Settings

In `r2-video-config.yaml`:

| Setting | Description |
|---------|-------------|
| `account_id` | Cloudflare Account ID (from R2 dashboard) |
| `bucket` | R2 bucket name (e.g., `argus-artifacts`) |
| `api_callback_url` | Optional: Backend API URL for metadata persistence |
| `access_key_id` | R2 API Token access key |
| `secret_access_key` | R2 API Token secret key |

### Resource Allocation

Adjust based on your workload:

| Component | Min Memory | Max Memory | Notes |
|-----------|------------|------------|-------|
| Browser Node | 512Mi | 1.5Gi | Main browser container |
| Video Recorder | 256Mi | 512Mi | ffmpeg sidecar |
| R2 Uploader | 64Mi | 128Mi | aws-cli sidecar |

## KEDA Integration

Update KEDA ScaledObjects to account for video recording overhead:

```yaml
# keda-scaledobject-with-video.yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: selenium-chrome-nodes-scaler
  namespace: selenium-grid
spec:
  scaleTargetRef:
    name: selenium-chrome-node
  minReplicaCount: 1
  maxReplicaCount: 50
  pollingInterval: 15
  cooldownPeriod: 120  # Longer cooldown for video upload completion

  triggers:
    - type: selenium-grid
      metadata:
        url: 'http://selenium-hub.selenium-grid.svc:4444/graphql'
        browserName: 'chrome'
        sessionBrowserName: 'chrome'
        activationThreshold: '0'
```

**Important:** Increase `cooldownPeriod` to allow video upload to complete before pod termination.

## Monitoring

### Check Upload Status

```bash
# Watch R2 uploader logs
kubectl logs -f -n selenium-grid -l app=selenium-chrome-node -c r2-uploader

# Check for failed uploads
kubectl logs -n selenium-grid -l app=selenium-chrome-node -c r2-uploader | grep "failed"
```

### Verify Videos in R2

```bash
# Using rclone (configure with R2 credentials first)
rclone ls r2:argus-artifacts/videos/

# Or via AWS CLI
aws s3 ls s3://argus-artifacts/videos/ \
  --endpoint-url https://<ACCOUNT_ID>.r2.cloudflarestorage.com
```

### Prometheus Metrics

Add to your monitoring:

```yaml
# Custom metrics for video recording
- job_name: 'selenium-video-uploader'
  kubernetes_sd_configs:
    - role: pod
  relabel_configs:
    - source_labels: [__meta_kubernetes_pod_container_name]
      regex: r2-uploader
      action: keep
```

## Troubleshooting

### Videos Not Uploading

1. **Check credentials:**
   ```bash
   kubectl get secret r2-credentials -n selenium-grid -o yaml
   # Verify base64 encoded values are correct
   ```

2. **Check R2 connectivity:**
   ```bash
   kubectl exec -it -n selenium-grid selenium-chrome-node-xxx -c r2-uploader -- \
     aws s3 ls s3://argus-artifacts/ \
     --endpoint-url https://<ACCOUNT_ID>.r2.cloudflarestorage.com
   ```

3. **Check video directory:**
   ```bash
   kubectl exec -it -n selenium-grid selenium-chrome-node-xxx -c r2-uploader -- \
     ls -la /videos/
   ```

### Video Quality Issues

1. **Increase frame rate:**
   ```yaml
   videoRecording:
     frameRate: 25  # Higher = smoother but larger files
   ```

2. **Increase resolution:**
   ```yaml
   videoRecording:
     screenSize: "1920x1080"  # or "2560x1440" for 2K
   ```

### R2 Permission Errors

Ensure your R2 API token has:
- Object Read permission
- Object Write permission
- Scoped to the correct bucket

## Cost Considerations

### R2 Pricing (as of 2025)

| Resource | Free Tier | Price After |
|----------|-----------|-------------|
| Storage | 10 GB/month | $0.015/GB/month |
| Class A Ops | 1M/month | $4.50/M |
| Class B Ops | 10M/month | $0.36/M |
| Egress | Unlimited! | FREE |

### Estimated Costs

| Scale | Videos/Day | Avg Size | Monthly Storage | Monthly Cost |
|-------|------------|----------|-----------------|--------------|
| Dev | 50 | 5MB | 7.5 GB | Free |
| Small | 500 | 5MB | 75 GB | ~$1.12 |
| Medium | 2,000 | 5MB | 300 GB | ~$4.50 |
| Large | 10,000 | 5MB | 1.5 TB | ~$22.50 |

**Note:** Zero egress fees makes R2 significantly cheaper than S3/GCS for video serving!

## Next Steps

1. **Enable signed URLs** - Configure `MEDIA_SIGNING_SECRET` for secure video access
2. **Set up retention policy** - Auto-delete old videos with R2 lifecycle rules
3. **Add alerting** - Monitor upload failures and disk space
4. **Implement cleanup** - Use R2 lifecycle policies for automatic video expiration

## References

- [Selenium Grid Video Recording](https://www.selenium.dev/documentation/grid/configuration/cli_options/#video-recording)
- [Cloudflare R2 Documentation](https://developers.cloudflare.com/r2/)
- [KEDA Selenium Grid Scaler](https://keda.sh/docs/scalers/selenium-grid-scaler/)
- [AWS CLI S3 Commands](https://docs.aws.amazon.com/cli/latest/reference/s3/)
