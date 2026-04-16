
#!/usr/bin/env bash
set -euo pipefail

REGISTRY="172.22.174.102:5000"
IMAGE_NAME="mlops-backend"
IMAGE_TAG="latest"
NAMESPACE="default"
DEPLOYMENT="backend"

echo "==> Repo root"
cd /root/MLOPS-Experiment02

echo "==> Activate venv if present"
if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi

#echo "==> Run full pipeline to refresh processed + weights artifacts"
#python main_pipeline.py

echo "==> Refresh inference bundle with latest mapping PKLs"
cp -v artifacts/processed/user2user_encoded.pkl artifacts/weights/
cp -v artifacts/processed/user2user_decoded.pkl artifacts/weights/
cp -v artifacts/processed/anime2anime_encoded.pkl artifacts/weights/
cp -v artifacts/processed/anime2anime_decoded.pkl artifacts/weights/

echo "==> Show artifact timestamps"
ls -lh --time-style=long-iso artifacts/weights/*.pkl artifacts/processed/*encoded*.pkl artifacts/processed/*decoded*.pkl

echo "==> Local artifact validation"
python - <<'PY'
import joblib, json, os

user_emb = joblib.load("artifacts/weights/user_weights.pkl")
anime_emb = joblib.load("artifacts/weights/anime_weights.pkl")
user2idx = joblib.load("artifacts/weights/user2user_encoded.pkl")
idx2user = joblib.load("artifacts/weights/user2user_decoded.pkl")
anime2idx = joblib.load("artifacts/weights/anime2anime_encoded.pkl")
idx2anime = joblib.load("artifacts/weights/anime2anime_decoded.pkl")

if isinstance(idx2user, list):
    idx2user = {i: v for i, v in enumerate(idx2user)}
elif not isinstance(idx2user, dict):
    idx2user = dict(idx2user)

if isinstance(idx2anime, list):
    idx2anime = {i: v for i, v in enumerate(idx2anime)}
elif not isinstance(idx2anime, dict):
    idx2anime = dict(idx2anime)

print("USER:", user_emb.shape[0], len(user2idx), len(idx2user))
print("ANIME:", anime_emb.shape[0], len(anime2idx), len(idx2anime))

assert user_emb.shape[0] == len(user2idx) == len(idx2user), "User artifacts mismatch"
assert anime_emb.shape[0] == len(anime2idx) == len(idx2anime), "Anime artifacts mismatch"
print("Artifact validation passed")
PY

echo "==> Build image"
docker build -t ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG} -f backend/Dockerfile .

echo "==> Push image"
docker push ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}

echo "==> Apply manifests"
kubectl apply -f deploy/redis.yaml
kubectl apply -f deploy/backend-deployment.yaml
kubectl apply -f deploy/prometheus-observability.yaml
kubectl apply -f deploy/grafana-datasource.yaml

echo "==> Restart backend deployment"
kubectl rollout restart deployment/${DEPLOYMENT} -n ${NAMESPACE}
kubectl rollout status deployment/${DEPLOYMENT} -n ${NAMESPACE}

echo "==> Show pod/service state"
kubectl get pods -n ${NAMESPACE} -o wide
kubectl get svc -A -o wide

echo "==> Backend logs"
kubectl logs deployment/${DEPLOYMENT} -n ${NAMESPACE} --tail=100

echo "==> Clear cached recommendations for test user"
TOKEN=$(curl -s -X POST http://backend-service.${NAMESPACE}.svc.cluster.local:8000/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=demo&password=demo" | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")
curl -s -X DELETE "http://backend-service.${NAMESPACE}.svc.cluster.local:8000/admin/cache/user/11880" \
  -H "Authorization: Bearer ${TOKEN}" || true

echo "==> Health check from cluster"
kubectl run curltest --rm -i --tty --image=curlimages/curl --restart=Never -- \
  curl -s http://backend-service.${NAMESPACE}.svc.cluster.local:8000/healthz
