# Model checkpoint (DVC + MinIO)

Бинарный чекпоинт **`model.pt`** (~1.1 GB) не хранится в Git — только метаданные в `model.pt.dvc`.
Артефакт лежит в MinIO: `s3://{{ cookiecutter.minio_bucket_models }}/dvc`.

## Первый запуск / новый клон

```bash
uv sync --group dev
docker compose --env-file .env.docker.compose up -d minio   # если MinIO ещё не поднят
./scripts/dvc-setup.sh      # endpoint + credentials из .env.docker.compose
dvc pull models/model.pt.dvc
```

Файл появится здесь: `models/model.pt` → монтируется в контейнер как `/models/model.pt`.

## Обновить версию в remote

После замены `models/model.pt` локально:

```bash
dvc add models/model.pt
git add models/model.pt.dvc
dvc push
git commit -m "chore(dvc): update model checkpoint"
```

## Проверка

```bash
dvc status
dvc remote list
```

MinIO Console: http://localhost:9001 → bucket `{{ cookiecutter.minio_bucket_models }}` → prefix `dvc/`.
