"""
Knowledge Graph Embedding - Hyperparameter Optimization
Runs grid search for multiple models and saves all metrics.
"""

import logging
from datetime import datetime
from pathlib import Path

import torch
from pykeen.hpo import hpo_pipeline
from pykeen.triples import TriplesFactory

from kg_utils import build_kg


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(log_file='hpo.log'):
    """Setup logging to both file and console."""
    Path(log_file).parent.mkdir(exist_ok=True)

    # Create logger
    logger = logging.getLogger('KGE_HPO')
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    logger.handlers = []

    # File handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# Initialize logger (will be set in main)
logger = None

# ============================================================================
# Configuration
# ============================================================================

MODELS = ['TransE', 'ComplEx', 'DistMult']
N_TRIALS = 15
OUTPUT_DIR = 'hpo_results'
RANDOM_SEED = 42


# ============================================================================
# Main HPO Function
# ============================================================================

def run_hpo_for_all_models(kg, models=MODELS, n_trials=N_TRIALS,
                           output_dir=OUTPUT_DIR, random_seed=RANDOM_SEED):
    """
    Run hyperparameter optimization for multiple KGE models.

    Args:
        kg: DataFrame with columns ['head', 'relation', 'tail']
        models: List of model names
        n_trials: Number of optimization trials per model
        output_dir: Directory to save results
        random_seed: Random seed for reproducibility
    """
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    # Create triples factory
    logger.info("Creating TriplesFactory from knowledge graph...")
    tf = TriplesFactory.from_labeled_triples(
        kg[['head', 'relation', 'tail']].values
    )

    logger.info("")
    logger.info("Knowledge Graph Statistics:")
    logger.info(f"  Entities: {tf.num_entities:,}")
    logger.info(f"  Relations: {tf.num_relations}")
    logger.info(f"  Triples: {tf.num_triples:,}")

    # Split data once (same split for all models - fair comparison)
    logger.info("")
    logger.info(f"Splitting data (80/10/10 train/test/val, seed={random_seed})...")
    training, testing, validation = tf.split([0.8, 0.1, 0.1], random_state=random_seed)

    logger.info(f"  Training triples: {training.num_triples:,}")
    logger.info(f"  Testing triples: {testing.num_triples:,}")
    logger.info(f"  Validation triples: {validation.num_triples:,}")

    # Run HPO for each model
    for model_idx, model_name in enumerate(models, 1):
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"Starting HPO for {model_name} ({model_idx}/{len(models)}) - {n_trials} trials")
        logger.info("=" * 70)
        logger.info("")

        start_time = datetime.now()

        try:
            logger.info(f"Running hyperparameter optimization...")
            hpo_result = hpo_pipeline(
                n_trials=n_trials,
                training=training,
                testing=testing,
                validation=validation,
                model=model_name,
                training_kwargs=dict(
                    num_epochs=1000,
                    use_tqdm_batch=False,
                ),
                stopper='early',
                stopper_kwargs=dict(
                    frequency=10,
                    patience=5,
                    relative_delta=0.002,
                ),
                metric='hits@10',
                device=torch.device('cuda'),
            )

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            logger.info(f"HPO completed in {duration:.1f}s")

            model_output_dir = Path(output_dir) / model_name
            model_output_dir.mkdir(parents=True, exist_ok=True)
            hpo_result.save_to_directory(model_output_dir)

        except Exception as e:
            print(f"\n✗ ERROR with {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue


# ============================================================================
# Execute
# ============================================================================

if __name__ == "__main__":
    import sys

    # Setup logging first
    logger = setup_logging(log_file='hpo_results/hpo.log')

    logger.info("=" * 70)
    logger.info("Knowledge Graph Embedding - Hyperparameter Optimization")
    logger.info("=" * 70)
    logger.info("")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Models to test: {', '.join(MODELS)}")
    logger.info(f"Trials per model: {N_TRIALS}")
    logger.info(f"Random seed: {RANDOM_SEED}")
    logger.info("")

    kg = build_kg(include_is_a=True)

    try:
        run_hpo_for_all_models(
            kg=kg,
            models=MODELS,
            n_trials=N_TRIALS,
            output_dir=OUTPUT_DIR,
            random_seed=RANDOM_SEED
        )

        logger.info("")
        logger.info("✓ All done! Review the results and choose your best model.")
        logger.info("  Then run the detailed training script with that model.")
        logger.info("")
        logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    except Exception as e:
        logger.error("")
        logger.error("=" * 70)
        logger.error("FATAL ERROR")
        logger.error("=" * 70)
        logger.error(str(e))
        import traceback

        logger.error(traceback.format_exc())
        sys.exit(1)
