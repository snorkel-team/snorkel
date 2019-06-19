import os

import torch


class Checkpointer(object):
    def __init__(self, config, verbose=True):
        """Saves checkpoints as applicable based on a reported metric.

        Args:
            checkpoint_runway (int): don't save any checkpoints for the first
                this many iterations
            checkpoint_dir (str): the directory for saving checkpoints
        """
        self.best_model_found = None
        self.best_iteration = None
        self.best_score = None
        self.verbose = verbose

        self.checkpoint_best = config["checkpoint_best"]
        self.checkpoint_every = config["checkpoint_every"]
        self.checkpoint_metric = config["checkpoint_metric"]
        self.checkpoint_metric_mode = config["checkpoint_metric_mode"]
        self.checkpoint_dir = config["checkpoint_dir"]
        self.checkpoint_runway = config["checkpoint_runway"]

        # Create checkpoint directory if necessary
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        # Remind about checkpoint runway
        if self.checkpoint_runway and verbose:
            print(
                f"No checkpoints will be saved in the first "
                f"checkpoint_runway={self.checkpoint_runway} iterations."
            )

    def checkpoint(self, metrics_dict, iteration, model, optimizer, lr_scheduler):
        # Return early if checkpoint_runway has not been met
        if self.checkpoint_runway:
            if iteration < self.checkpoint_runway:
                return
            elif iteration == self.checkpoint_runway:
                print("Checkpoint runway has been met. Checkpointing will now occur.")

        if (
            self.checkpoint_every
            and iteration > 0
            and iteration % self.checkpoint_every == 0
        ):
            # Save the checkpoint regardless of performance
            score = None
            state = self.bundle_state(iteration, score, model, optimizer, lr_scheduler)
            checkpoint_path = f"{self.checkpoint_dir}/model_checkpoint_{iteration}.pth"
            torch.save(state, checkpoint_path)

        if self.checkpoint_best and self.checkpoint_metric in metrics_dict:
            score = metrics_dict[self.checkpoint_metric]
            if self.is_best(score):
                if self.verbose:
                    print(
                        f"Saving model at iteration {iteration:.2f} with best "
                        f"({self.checkpoint_metric_mode}) score "
                        f"{self.checkpoint_metric}={score:.3f}"
                    )
                self.best_model_found = True
                self.best_iteration = iteration
                self.best_score = score

                # Save the checkpoint, overriding previous best if it exists
                state = self.bundle_state(
                    iteration, score, model, optimizer, lr_scheduler
                )
                checkpoint_path = f"{self.checkpoint_dir}/best_model.pth"
                torch.save(state, checkpoint_path)

    def is_best(self, score):
        if self.best_score is None:
            return True
        elif self.checkpoint_metric_mode == "max":
            return score > self.best_score
        elif self.checkpoint_metric_mode == "min":
            return score < self.best_score
        else:
            msg = (
                f"Did not recognize checkpoint_metric_mode: "
                + f"{self.checkpoint_metric_mode}"
            )
            raise ValueError(msg)

    def bundle_state(self, iteration, score, model, optimizer, lr_scheduler):
        # Save the state of the best model
        state = {
            "iteration": iteration,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler else None,
            "score": score,
        }
        if self.best_model_found:
            state["best_model_found"] = True
            state["best_iteration"] = self.best_iteration
            state["best_score"] = self.best_score
        return state

    def load_best_model(self, model):
        if self.best_model_found is None:
            msg = (
                f"Best model was never found. Confirm that your checkpoint_metric "
                f"({self.checkpoint_metric}) is of the form "
                f"'[model or task]/[split]/loss' or produced by one of your tasks' "
                f"Scorers and that checkpoint_metric_mode "
                f"({self.checkpoint_metric_mode}) is appropriate for the given "
                f"checkpoint_metric."
            )
            raise Exception(msg)
        if self.verbose:
            print(
                f"Restoring best model from iteration {self.best_iteration:0.2f} "
                f"with score {self.best_score:.3f}"
            )
            state = torch.load(
                f"{self.checkpoint_dir}/best_model.pth",
                map_location=torch.device("cpu"),
            )
            self.best_iteration = state["best_iteration"]
            self.best_score = state["best_score"]
            model.load_state_dict(state["model"])
            return model

    def restore(self, destination):
        state = torch.load(f"{destination}")
        return state

    def clean_up(self):
        if os.path.exists(self.checkpoint_dir):
            os.system(f"rm -r {self.checkpoint_dir}")
