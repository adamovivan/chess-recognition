gcloud ml-engine jobs submit training JOB37 --module-name=trainer.cnn_with_keras --package-path=./trainer --job-dir=gs://chess-recognition-bucket --region=us-east1 --config=trainer/cloudml-gpu.yaml
