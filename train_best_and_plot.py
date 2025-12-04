import pickle

import pandas as pd
import torch
from matplotlib import pyplot as plt
from pykeen.evaluation import RankBasedEvaluator
from pykeen.pipeline import pipeline
from pykeen.training import TrainingCallback
from pykeen.triples import TriplesFactory


class MetricTracker(TrainingCallback):
    """Track training and validation metrics during training."""

    @property
    def model(self):
        return self._model

    def __init__(self, validation_triples, eval_frequency=10):
        super().__init__()
        self.validation_triples = validation_triples
        self.eval_frequency = eval_frequency
        self.evaluator = RankBasedEvaluator()

        self.batch_losses = []
        self.batch_epochs = []
        self.epoch_data = []
        self.val_data = []
        self.model = None

    def register_training_loop(self, training_loop):
        """Called by PyKEEN to give callback access to the training loop."""
        super().register_training_loop(training_loop)
        self.model = training_loop.model

    def on_batch(self, epoch: int, batch, batch_loss: float, **kwargs):
        """Called after each batch - has access to batch_loss."""
        self.batch_losses.append(float(batch_loss))
        self.batch_epochs.append(epoch)

    def post_epoch(self, epoch: int, epoch_loss: float, **kwargs):
        """Called after each epoch."""
        self.epoch_data.append({
            'epoch': epoch,
            'train_loss': float(epoch_loss)
        })

        # Evaluate on validation set
        if epoch % self.eval_frequency == 0 and self.model is not None:
            self.model.eval()
            with torch.no_grad():
                val_results = self.evaluator.evaluate(
                    model=self.model,
                    mapped_triples=self.validation_triples.mapped_triples,
                    batch_size=256,
                    use_tqdm=False,
                )
            self.model.train()

            self.val_data.append({
                'epoch': epoch,
                'hits@10': float(val_results.get_metric('hits@10'))
            })

    @model.setter
    def model(self, value):
        self._model = value


def train_best_model():
    # Load data
    kg = pickle.load(open("./data/is_associated_with_only.pkl", "rb"))
    tf = TriplesFactory.from_labeled_triples(
        kg[['head', 'relation', 'tail']].values
    )
    training, testing, validation = tf.split([0.8, 0.1, 0.1], random_state=42)

    # Create tracker
    tracker = MetricTracker(validation_triples=validation, eval_frequency=10)

    # Train model
    result = pipeline(
        training=training,
        testing=testing,
        validation=validation,
        model="TransE",
        model_kwargs=dict(
            embedding_dim=144,
        ),
        training_kwargs=dict(
            num_epochs=200,
            batch_size=2048,
            use_tqdm_batch=False,
            callbacks=[tracker]
        ),
        optimizer_kwargs=dict(
            lr=1e-3,
        ),
        stopper='early',
        stopper_kwargs=dict(
            frequency=10,
            patience=5,
            relative_delta=0.002,
            metric='hits@10',
        ),
        random_seed=42,
    )

    # Get dataframes
    epoch_df = pd.DataFrame(tracker.epoch_data)
    val_df = pd.DataFrame(tracker.val_data)

    print(f"\nFinal test results:")
    print(f"  Hits@10: {result.metric_results.get_metric('hits@10'):.4f}")
    print(f"  MRR: {result.metric_results.get_metric('mrr'):.4f}")

    return epoch_df, val_df, result


def plot_best_model(epoch_df, val_df):
    """Plot training loss vs validation hits@10."""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Training loss
    ax1.plot(epoch_df['epoch'], epoch_df['train_loss'],
             label="Train loss", color="tab:blue", linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Training loss", color="tab:blue", fontsize=12)
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.3)

    # Validation hits@10
    ax2 = ax1.twinx()
    ax2.plot(val_df['epoch'], val_df['hits@10'],
             "o-", label="Val hits@10", color="tab:orange", linewidth=2, markersize=6)
    ax2.set_ylabel("Validation hits@10", color="tab:orange", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax2.set_ylim(0, 1.0)

    plt.title("TransE: Training Loss vs. Validation Hits@10", fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.savefig('best_model_plot.png', dpi=300, bbox_inches='tight')
    print("\n✓ Plot saved to best_model_plot.png")
    plt.close()


if __name__ == '__main__':
    print("Training best model...")
    epoch_df, val_df, result = train_best_model()

    print("\nCreating plot...")
    plot_best_model(epoch_df, val_df)

    print("\n✓ Done!")
