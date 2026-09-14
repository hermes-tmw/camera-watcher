# Build + test entrypoints for camera-watcher.
#
# `hermes verify` detects this Makefile (its docker-compose fallback recipe
# builds only the default services and has NO test phase) and runs:
#   build -> make build   (all service images incl. stealthcam/wichita/classifier)
#   test  -> make test    (stealthcam suite, stdlib unittest)
# There is deliberately no run/start/serve/dev target: `docker compose up`
# boots REAL workers against the production database, so machine verification
# stops at build+test; starting the stack is an operator decision.

.PHONY: build test

# Two steps, in order: `FROM camera-watcher-io` (Dockerfile-video) resolves the
# local tag only if it already exists — compose builds services in parallel and
# does NOT order by depends_on (verified empirically), so a virgin machine's
# single parallel build can fail trying to PULL camera-watcher-io from Docker
# Hub. Building io first guarantees the tag, then --profile all builds the rest.
build:
	docker compose build io
	docker compose --profile all build

# WATCHER_CONFIG points at the committed example so the suite runs on a clean
# clone with no local secret files. Prereqs: python3 with the requirements-io
# + requirements-stealthcam deps (flask, sqlalchemy, psycopg2, ultralytics...).
test:
	WATCHER_CONFIG=etc/watcher.cfg.example python3 -m unittest discover -s stealthcam/tests