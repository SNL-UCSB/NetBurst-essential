"""Continue training from a saved NetBurst checkpoint directory (``TwinHeadChronosPredictor.from_pretrained``)."""

from netburst.train_loop import run_training

if __name__ == "__main__":
    run_training(from_checkpoint=True)
