# OxBlue Site Analyzer — Azure Web App Deployment

## Pre-deploy: bundle the YOLO models

The models must be in the zip so Azure doesn't need to download them on
every cold start (slow + unreliable on restricted networks).

```bash
# Run once locally — downloads models to your project folder
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
python -c "
import urllib.request
urllib.request.urlretrieve(
  'https://huggingface.co/keremberke/yolov8n-hard-hat-detection/resolve/main/best.pt',
  'hard_hat_yolov8n.pt'
)
print('Done')
"
```

Your folder should now contain:
```
server.py
index.html
oxblue-logo.png
requirements.txt
startup.sh
web.config
yolov8n.pt           ← bundled model
hard_hat_yolov8n.pt  ← bundled model
```

---

## Deploy with Azure CLI

```bash
# 1. Login
az login

# 2. Create resource group (skip if you have one)
az group create --name oxblue-rg --location eastus

# 3. Create App Service plan (B2 = 3.5GB RAM, needed for YOLO)
az appservice plan create \
  --name oxblue-plan \
  --resource-group oxblue-rg \
  --sku B2 \
  --is-linux

# 4. Create the Web App
az webapp create \
  --name oxblue-analyzer \
  --resource-group oxblue-rg \
  --plan oxblue-plan \
  --runtime "PYTHON:3.11"

# 5. Set startup command
az webapp config set \
  --name oxblue-analyzer \
  --resource-group oxblue-rg \
  --startup-file "bash startup.sh"

# 6. Set your Anthropic API key
az webapp config appsettings set \
  --name oxblue-analyzer \
  --resource-group oxblue-rg \
  --settings ANTHROPIC_API_KEY=sk-ant-your-key-here

# 7. Increase upload size limit (videos can be large)
az webapp config appsettings set \
  --name oxblue-analyzer \
  --resource-group oxblue-rg \
  --settings WEBSITES_MAX_REQUEST_SIZE=500

# 8. Zip and deploy
zip -r deploy.zip . \
  --exclude "*.pyc" \
  --exclude "__pycache__/*" \
  --exclude ".git/*" \
  --exclude "deploy.zip"

az webapp deployment source config-zip \
  --name oxblue-analyzer \
  --resource-group oxblue-rg \
  --src deploy.zip

# 9. Open in browser
az webapp browse --name oxblue-analyzer --resource-group oxblue-rg
```

Your app will be live at:
https://oxblue-analyzer.azurewebsites.net

---

## Important settings to configure in Azure Portal

Go to your Web App → Configuration → General Settings:

| Setting | Value |
|---|---|
| HTTP version | 2.0 |
| ARR Affinity | Off (single worker anyway) |
| Always On | On (B2 and above support this) |

Go to Configuration → Application Settings and confirm:
- `ANTHROPIC_API_KEY` is set
- `SCM_DO_BUILD_DURING_DEPLOYMENT` = `true`

---

## Troubleshooting

**App won't start / 503 errors**
```bash
az webapp log tail --name oxblue-analyzer --resource-group oxblue-rg
```

**Video upload fails (413 error)**
Add this app setting:
```bash
az webapp config appsettings set \
  --name oxblue-analyzer \
  --resource-group oxblue-rg \
  --settings WEBSITES_MAX_REQUEST_SIZE=500
```

**Analysis times out**
The default Azure timeout is 230 seconds. For long videos, either:
- Reduce sample rate to 1 frame / 2 sec in the UI
- Upgrade to P1v3 plan which supports longer timeouts

**Models re-downloading on every restart**
Make sure `yolov8n.pt` and `hard_hat_yolov8n.pt` are in your zip file.
