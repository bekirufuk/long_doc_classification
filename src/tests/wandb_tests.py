import wandb
api = wandb.Api()

run = api.run("bekirufuk/Long Document Classification/2rvylwyk")
#run.summary["accuracy"] = 0.9
#run.summary["accuracy_histogram"] = wandb.Histogram(numpy_array)
#run.summary.update()
#run.summary['Test Accuracy'] = 0.7522
#run.summary.update()