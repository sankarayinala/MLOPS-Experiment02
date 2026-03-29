# worker/celery_beat.py
from celery.schedules import crontab

app.conf.beat_schedule = {
    "retrain-every-day-2am": {
        "task": "worker.celery_worker.retrain_models",
        "schedule": crontab(hour=2, minute=0),
    },
}