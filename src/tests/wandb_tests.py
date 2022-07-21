import wandb
api = wandb.Api()

run = api.run("bekirufuk/Long Document Classification/1lryg03m")
run.summary['Test Accuracy'] = 0.7578
run.summary.update()