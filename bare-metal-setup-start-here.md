# pythia — Bare-Metal Setup (Start Here)

Stand up **pythia** from scratch on a fresh **Kubuntu** box, reproducing the production stack
(single-node **k3s**). Pythia is a standalone probabilistic-forecasting service: a read-only
**serve** Deployment (a tiny FastAPI over a Postgres model registry) plus a **nightly-retrain**
CronJob that assembles a dataset, runs a walk-forward TFT-lite backtest, and registers the report.
This guide is self-contained: everything you need to go from a bare OS to a verified-running app is
here. Where pythia depends on a neighboring system (the **quote/registry Postgres**, an **NVIDIA
GPU** for retrain, the raptor-intel **frontend** that renders its forecasts), you'll see an
**[external]** marker and either how to point at an existing one or how to stand up "just enough".

> Prod reality this mirrors: cluster **achilles k3s**, namespace `pythia`. **One** app Deployment
> (`pythia-serve`, 1 replica) + **one** CronJob (`pythia-nightly-retrain`, `15 9 * * 1-5` UTC) + a
> 10Gi PVC (`pythia-data`) for dataset/report scratch. **No in-cluster database of its own** — it
> reuses raptor's Postgres. **Serve needs no GPU; the retrain CronJob needs one.**

---

## 0. Prerequisites

- A fresh Kubuntu box (22.04+), a sudo user, and internet access.
- Serve alone is tiny (requests `50m` CPU / `128Mi`; limit `1` CPU / `512Mi`). The **retrain**
  CronJob needs **~2 vCPU / 4 GB RAM + 1 NVIDIA GPU**; add ~10 GB disk for the PVC + images.
- The `tonyv2020/pythia` repo. **Clone it to the exact host path the retrain CronJob mounts:**
  `sudo git clone https://github.com/tonyv2020/pythia /home/shared/dev/workspace/pythia`
  (⚠ the CronJob hostPath is hard-coded to `/home/shared/dev/workspace/pythia` — see §7).
- Python layout: `src/pythia` (hatchling wheel, `requires-python >=3.11`); serve image is
  `python:3.11-slim`, retrain runs on a stock `pytorch/pytorch` CUDA image.
- **Decide your target now:** *local-only* (serve + registry only, skip the GPU retrain and the
  external frontend — port-forward the API) vs *full* (GPU nightly retrain + consumed through
  raptor-intel's edge).

**[external] accounts/infra to have BEFORE you start** (only what your target needs):
- **A Postgres reachable** that holds (a) raptor's `staging.quote_raw` quote history — the retrain's
  **training-data source** — and (b) the `pythia_models` registry table (auto-created; can be the
  **same** database). Prod points all DSNs at raptor's always-on Postgres
  (`postgres.hollywood.svc.cluster.local:5432/raptor`). Without it, serve returns 404 ("no model
  registered") and retrain cannot run — see §6. **Required.**
- *(retrain only)* an **NVIDIA GPU** on the node with the k3s `nvidia` RuntimeClass working
  (`runtimeClassName: nvidia`, `nvidia.com/gpu: 1`). Serve does not need it.
- *(consumption only)* a running **raptor-intel** whose frontend nginx reverse-proxies
  `/pythia/api/*` → `pythia-serve` (§8). Pythia has **no public domain of its own**.

---

## 1. Base system + container runtime

```bash
sudo apt-get update && sudo apt-get install -y ca-certificates curl git jq
# Docker (for building the serve image)
curl -fsSL https://get.docker.com | sudo sh && sudo usermod -aG docker "$USER"   # re-login after
# k3s single-node
curl -sfL https://get.k3s.io | sh -
sudo k3s kubectl get nodes    # confirm Ready
# convenient kubeconfig for your user
mkdir -p ~/.kube && sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config \
  && sudo chown "$USER" ~/.kube/config && sed -i 's/127.0.0.1/'"$(hostname -I|awk '{print $1}')"'/' ~/.kube/config
```

> **[external] GPU (retrain only):** the nightly CronJob schedules on `nvidia.com/gpu: 1` under the
> `nvidia` RuntimeClass. On a fresh box that means: NVIDIA driver installed, the
> `nvidia-device-plugin` DaemonSet running, and an `nvidia` RuntimeClass registered (k3s +
> nvidia-container-runtime). If you're **serve-only / local-only**, skip all of this — serve is
> pure CPU. (Reference: helen's GPU-recovery runbook — after a kernel bump you reinstall
> `linux-modules-nvidia-*-$(uname -r)`, `modprobe`, restart the device-plugin.)

> ⚠ **No build/import script in the repo.** Unlike raptor (`scripts/build-and-import-images.sh`),
> pythia ships **no** image-build helper. The serve image is built by hand from
> `docker/Dockerfile.serve` and the manifest pulls `registry.lan/pythia-serve:<tag>`
> (`imagePullPolicy: IfNotPresent`). On a single box you have two equivalent choices — push to a LAN
> registry, or build and `k3s ctr images import` straight into containerd (§5).

---

## 2. Namespace + storage

`k8s/00-namespace.yaml` creates the namespace **and** the scratch PVC in one file:

```bash
kubectl apply -f k8s/00-namespace.yaml
# creates: Namespace pythia + PVC pythia-data (RWO, 10Gi, default storageclass = local-path)
kubectl -n pythia get pvc pythia-data      # expect Bound
```

The PVC is **scratch space** for the retrain (`/data/board.parquet`, `/data/report.json`), not a
database. Pythia has **no Postgres/Neo4j/Redis StatefulSet of its own** — the model registry and the
training data both live in raptor's Postgres (§6). There is **no in-namespace ConfigMap** (prod ns
has only the auto `kube-root-ca.crt`); all config is env, injected inline + from the one secret (§3–4).

---

## 3. Config

There is no ConfigMap. Serve config is two env vars set in `k8s/serve-deployment.yaml`:

- `PYTHIA_REGISTRY_DSN` — from secret `pythia-db` (the DB it **reads** registered models from).
  ⚠ **Redundant keys, NOT a live bug (author-confirmed):** the serve manifest wires this from key
  **`dsn`** while the retrain CronJob uses **`registry_dsn`** — but in prod both point at the **same**
  registry DB (`appdb`), so serve and the nightly agree and `/latest` works. It is a *consistency
  smell*, not a break: if someone ever repoints `dsn` (e.g. to a read-only replica) without also
  updating `registry_dsn`, serve would silently read a split/stale registry and re-orphan the P5
  blocks. **Hardening (low priority): unify serve on `registry_dsn`.** Keep `dsn` ≡ `registry_dsn`
  unless you deliberately split roles.
- `PYTHIA_CORS_ORIGINS` — inline literal `https://raptor.tonyvigna.com,https://tonyvigna.com`
  (GET-only CORS). **Edit this** to your own origin(s) or the panel's `fetch` is blocked.

The retrain CronJob sets two DSN env vars (§7): `PYTHIA_DB_DSN` (from `src_dsn` — the **training-data
source**) and `PYTHIA_REGISTRY_DSN` (from `registry_dsn` — where the report is **written**).

DSN resolution in code (`src/pythia/config.py`, `registry/models.py`): env wins, else the in-cluster
default `postgresql://hollywood@postgres.hollywood.svc.cluster.local:5432/raptor`, table
`staging.quote_raw`. Registry DSN falls back `PYTHIA_REGISTRY_DSN → PYTHIA_DB_DSN → default`.

---

## 4. Secrets (create imperatively — values are yours)

⚠ **No example secret file in the repo.** There is exactly one secret, `pythia-db`, with **three
DSN keys**; you must mint it yourself. In prod all three point at raptor's Postgres — they can be the
same DSN, or differ if you want a read-only training role vs. a read-write registry role.

```bash
# All three keys carry a full SQLAlchemy/psycopg DSN, e.g.
#   postgresql://<user>:<pw>@<host>:5432/raptor
kubectl -n pythia create secret generic pythia-db \
  --from-literal=dsn='postgresql://<ro-or-rw>@<host>:5432/raptor' \
  --from-literal=src_dsn='postgresql://<training-read>@<host>:5432/raptor' \
  --from-literal=registry_dsn='postgresql://<registry-rw>@<host>:5432/raptor'
```

Key roles:
- **`dsn`** — read by `pythia-serve` (via `PYTHIA_REGISTRY_DSN` in the repo manifest) to serve the
  latest registered model. Read-only is fine.
- **`src_dsn`** — the retrain's `PYTHIA_DB_DSN`: reads `staging.quote_raw` for training data
  ([external], §6). Read-only is fine.
- **`registry_dsn`** — the retrain's `PYTHIA_REGISTRY_DSN`: **writes** the `pythia_models` row.
  Needs INSERT/UPDATE (and CREATE the first time, unless a DBA pre-creates the table — see §6).

> ⚠ **Tag drift, verify before applying.** The repo manifest pins `registry.lan/pythia-serve:0.1.0`;
> the live Deployment runs `:0.1.6`. Build/tag whatever you deploy and reconcile the manifest — don't
> assume `:0.1.0` is current (§5).

---

## 5. Build the serve image + deploy

```bash
# Build the read-only FastAPI serve image (python:3.11-slim; installs pythia + fastapi + sqlalchemy + psycopg).
docker build -t registry.lan/pythia-serve:0.1.6 -f docker/Dockerfile.serve .

# --- Option A: LAN registry (prod parity) ---
docker push registry.lan/pythia-serve:0.1.6
# --- Option B: single box, no registry (simpler, equivalent) ---
docker save registry.lan/pythia-serve:0.1.6 | sudo k3s ctr images import -

# Deploy serve (edit the image tag in the manifest to match what you built — see §4 drift note).
kubectl apply -f k8s/serve-deployment.yaml
kubectl -n pythia rollout status deploy/pythia-serve
```

`serve-deployment.yaml` creates the Deployment (`replicas: 1`, container port 8000, `/health`
readiness+liveness probes) **and** a `ClusterIP` Service `pythia-serve` on port **80 → 8000**.

> **What serve does:** read-only. It has no model files — every route (`/health`, `/latest`,
> `/attention`, `/variable-importance`, `/breakouts`, `/events`) reads the latest `pythia_models` row
> from Postgres and returns its stored walk-forward `report_json` verbatim (plus a derived
> calibration verdict). **A brand-new install with an empty registry returns `404 "no model
> registered under 'tft_lite_daily_qqq'"` on `/latest` until the retrain (§7) has run at least once.**

---

## 6. [external] The data store + training-data source

Pythia runs **no database**. Both the training data and the model registry live in **raptor's
Postgres** (prod: `postgres.hollywood.svc.cluster.local:5432/raptor`):

- **Training data (read):** table **`staging.quote_raw`** — raptor's intraday quote ticks for the
  20-symbol macro board (QQQ + 19 covariates; see `src/pythia/config.py`). The assembler rolls ticks
  into daily OHLCV bars; the nightly D8 backfill additionally pulls multi-year history from
  **yfinance** (2018-01-01 onward) and merges the raptor feed for recent bars. Point `src_dsn` here.
- **Model registry (read+write):** table **`pythia_models`** —
  `(model_name, model_version, trained_at, dataset_hash, report_json JSONB, artifact_uri, git_sha)`,
  UNIQUE `(model_name, model_version)`. `ModelRegistry.ensure_schema()` **auto-creates** it
  (idempotent `CREATE TABLE IF NOT EXISTS` + index). ⚠ If your registry role lacks CREATE, it tolerates
  that **only if the table already exists** — so either grant CREATE once, or have a DBA create
  `pythia_models` out-of-band. The retrain UPSERTs (`ON CONFLICT … DO UPDATE`), so reruns refresh
  numbers without inflating rows. Point `registry_dsn` here.

To point at your own database, just set the DSNs in `pythia-db` (§4) — nothing else changes. Without
a populated `staging.quote_raw`, the retrain has no data to train on; without `pythia_models`, serve
has nothing to return.

> The upstream that fills `staging.quote_raw` (E*TRADE → GCP Pub/Sub → raptor's consumer/worker) is a
> **separate system** owned by the raptor stack — pythia never touches it directly.

---

## 7. The nightly-retrain CronJob ([external] GPU)

```bash
kubectl apply -f k8s/nightly-retrain-cronjob.yaml
kubectl -n pythia get cronjob pythia-nightly-retrain      # SCHEDULE 15 9 * * 1-5, TZ UTC
# force a one-off run to seed the registry now (don't wait for 09:15 UTC):
kubectl -n pythia create job --from=cronjob/pythia-nightly-retrain pythia-manual-1
kubectl -n pythia logs -f job/pythia-manual-1
```

What it does (one execution of `scripts.nightly_retrain`): assemble the D8-backfilled full-board
dataset → walk-forward TFT-lite backtest (`scripts.train_p1_tft`, encoder=60/hidden=16) → register
the report in `pythia_models` as `v<UTC-date>` → soft-attach the P5a range + P5b breakout blocks.
Schedule `15 9 * * 1-5` = 09:15 UTC Mon–Fri (post-US-close so raptor's daily bars are final).
`concurrencyPolicy: Forbid`, `backoffLimit: 1`.

> ⚠ **This CronJob is NOT a self-contained image.** It runs the stock
> `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` image and **hostPath-mounts the repo checkout** at
> `/home/shared/dev/workspace/pythia` (→ `/pythia`), then does `pip install -e . scipy yfinance
> psycopg[binary]` inline before `python -m scripts.nightly_retrain`. Consequences for a fresh box:
> - The repo **must exist at exactly `/home/shared/dev/workspace/pythia` on the node** (§0), or the
>   pod fails to mount. Edit the `hostPath` in the manifest if you cloned elsewhere.
> - Because it mounts source (not a baked image), a `git pull`/merge on that checkout is picked up on
>   the **next** cron fire — no image rebuild for pythia code changes.
> - It requires the **NVIDIA GPU** (`runtimeClassName: nvidia`, `nvidia.com/gpu: 1`, §1). No GPU →
>   the pod stays Pending.

> ⚠ **Ignore `k8s/intraday-backtest-job.yaml` on a fresh box.** It's a one-shot P3 artifact that
> references `registry.lan/pythia-trainer:0.2.0` — a `pythia-trainer` image that **never existed in
> the registry** (the reason an earlier retrain CronJob never fired, fixed by switching to the
> pytorch-base + hostPath pattern above). Don't rely on it; the nightly path is the supported one.

---

## 8. [external] Exposure (how forecasts are consumed)

Pythia has **no ingress, no tunnel, no public domain**. It's an in-cluster `ClusterIP` only. It's
reached exactly one way in prod: **raptor-intel's frontend nginx reverse-proxies `/pythia/api/*`**,
stripping the prefix, to `http://pythia-serve.pythia.svc.cluster.local:80`
(`raptor-intel/frontend/nginx.conf`):

```nginx
location /pythia/api/ {
    rewrite ^/pythia/api/(.*)$ /$1 break;          # /pythia/api/latest -> /latest
    proxy_pass http://pythia-serve.pythia.svc.cluster.local:80;
    ...
}
```

So a browser hitting `https://raptor.tonyvigna.com/pythia/api/latest` lands on serve's `/latest`.
`PYTHIA_CORS_ORIGINS` (§3) must include whatever origin the panel is served from.

**For local-only** (no raptor), just port-forward and hit the API directly:

```bash
kubectl -n pythia port-forward svc/pythia-serve 8000:80
curl -s localhost:8000/health           # {"status":"ok","service":"pythia"}
curl -s 'localhost:8000/latest?model=tft_lite_daily_qqq' | jq .   # 404 until a retrain has run
```

---

## 9. Verify (the "it works" gate)

```bash
# serve health (in-pod)
kubectl -n pythia exec deploy/pythia-serve -- python -c \
  "import urllib.request,sys; print(urllib.request.urlopen('http://localhost:8000/health').read())"
# or via the Service:
kubectl -n pythia port-forward svc/pythia-serve 8000:80 &
curl -s localhost:8000/health                          # {"status":"ok","service":"pythia"}

# registry populated? (after at least one retrain in §7)
curl -s 'localhost:8000/latest?model=tft_lite_daily_qqq' | jq '{model_version,calibrated,notes}'
#   expect a v<date> version, a calibration verdict, and derived notes.
#   Header X-Pythia-Calibrated: true|false reflects the 0.75-0.85 P10-P90 coverage gate.

# retrain ran clean?
kubectl -n pythia get jobs
kubectl -n pythia logs job/<latest-retrain-job> --tail=20     # ends "nightly retrain complete: <date>"
```

The end-to-end gate: a retrain Job **Completed**, a row in `pythia_models`, and `/latest` returning
that row's report through the raptor panel (or the port-forward).

---

## 10. Teardown / reset

```bash
kubectl delete ns pythia                 # removes serve + cronjob + the PVC
# NOTE: the pythia_models table lives in raptor's Postgres — deleting the ns does NOT drop it.
#   To fully reset: psql "<registry_dsn>" -c 'DROP TABLE IF EXISTS pythia_models;'
# full box reset:  /usr/local/bin/k3s-uninstall.sh
```

---

## Prereqs summary (what a fresh operator must supply)

| Need | For | How |
|---|---|---|
| Postgres with `staging.quote_raw` (populated) | retrain training data | §4 `src_dsn` + §6; [external] raptor DB — required |
| Postgres for `pythia_models` (CREATE or pre-made) | model registry serve reads | §4 `registry_dsn`/`dsn` + §6; can be the same DB |
| `pythia-db` secret (keys `dsn`,`src_dsn`,`registry_dsn`) | all DSN wiring | §4; ⚠ no example file in repo — mint it |
| `pythia-serve` image built + reachable | serve Deployment | §5; ⚠ no build script — build by hand, push to `registry.lan` or `k3s ctr import` |
| NVIDIA GPU + `nvidia` RuntimeClass | nightly retrain only | §1/§7; skip for serve-only/local |
| Repo checkout at `/home/shared/dev/workspace/pythia` on the node | retrain hostPath mount | §0/§7; ⚠ hard-coded path — clone there or edit manifest |
| raptor-intel frontend (nginx `/pythia/api/*`) | consuming forecasts | §8; optional — port-forward locally instead |

*No GPU for serve. No Redis. No Neo4j. No database of pythia's own. No public domain (reached via
raptor's edge). No OIDC/SSO on pythia itself (the API is GET-only, CORS-gated, behind raptor).*

⚠ **Open gaps a fresh operator will hit:** (1) no image-build/import script; (2) serve image tag
drift (manifest `:0.1.0` vs live `:0.1.6`); (3) `registry.lan` LAN-registry assumption; (4) retrain's
hard-coded hostPath + GPU + inline `pip install` (not a baked image); (5) the dead
`intraday-backtest-job.yaml` referencing a never-existent `pythia-trainer` image; (6) no example
secret and three DSN keys with a subtle key-name mismatch (`dsn` in the serve manifest vs
`registry_dsn` live).
