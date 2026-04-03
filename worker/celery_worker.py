# worker/celery_worker.py

@app.task
def retrain_models():
    from pipeline.model_training import ModelTrainer
    trainer = ModelTrainer()
    trainer.run()