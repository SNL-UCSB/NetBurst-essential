"""Cold-start distributed NetBurst training (same CLI as legacy ``main``)."""

from netburst.train_loop import run_training

if __name__ == "__main__":
    run_training(from_checkpoint=False)
