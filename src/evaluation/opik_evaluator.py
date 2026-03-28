import os
import sys
from typing import Dict, Any, List, Optional

from opik import Opik
from opik.evaluation import evaluate
from opik.evaluation.metrics import Hallucination

from logger import get_logger
from custom_exception import CustomException

logger = get_logger(__name__)


class OpikEvaluator:
    """
    Clean and robust Opik evaluator for your recommendation model.
    """

    def __init__(
        self,
        experiment_name: str = "anime-recommender-eval-v1",
        dataset_name: str = "anime-recommendation-eval"
    ):
        self.experiment_name = experiment_name
        self.dataset_name = dataset_name

        # Set Opik credentials
        os.environ["OPIK_API_KEY"] = "kA1fB80gcoco5B1tsjnM18zSf"
        os.environ["OPIK_WORKSPACE"] = "sankar-bmc"

        try:
            self.client = Opik()
            logger.info(f"✅ Opik client initialized | Workspace: sankar-bmc")
        except Exception as e:
            logger.error(f"Failed to initialize Opik: {e}")
            raise CustomException("Opik initialization failed", e) from e

    def get_or_create_dataset(self):
        """Get or create the evaluation dataset"""
        try:
            dataset = self.client.get_dataset(name=self.dataset_name)
            logger.info(f"✅ Dataset found: '{self.dataset_name}'")
            return dataset
        except Exception:
            logger.warning(f"Dataset '{self.dataset_name}' not found. Creating new one...")

            # Sample data for initial testing
            sample_items = [
                {
                    "input": "Recommend some action anime with great fights",
                    "context": "User likes high-intensity battles and supernatural themes",
                    "expected_output": "Jujutsu Kaisen, Demon Slayer, Attack on Titan"
                },
                {
                    "input": "I want a heartfelt romance anime",
                    "context": "Emotional story, character development",
                    "expected_output": "Your Lie in April, A Silent Voice, Horimiya"
                },
                {
                    "input": "Suggest sci-fi anime with deep philosophy",
                    "context": "Mind-bending plots",
                    "expected_output": "Steins;Gate, Psycho-Pass"
                }
            ]

            dataset = self.client.create_dataset(
                name=self.dataset_name,
                description="Evaluation dataset for anime recommendation system"
            )

            # Insert items correctly
            for item in sample_items:
                dataset.insert(
                    input=item["input"],
                    expected_output=item.get("expected_output", ""),
                    context=item.get("context", "")
                )

            logger.info(f"✅ New dataset created: '{self.dataset_name}' with {len(sample_items)} items")
            return dataset

    def evaluation_task(self, dataset_item: Dict[str, Any]) -> Dict[str, Any]:
        """Main evaluation task - this is where your model runs"""
        try:
            user_input = dataset_item.get("input", "")
            context = dataset_item.get("context", "")

            # TODO: Replace this with your actual trained model inference
            output = self._generate_recommendation(user_input, context)

            return {
                "input": user_input,
                "output": output,
                "context": context,
                "metadata": {"model": "recommender-net"}
            }

        except Exception as e:
            logger.error(f"Error in evaluation task: {e}")
            return {
                "input": dataset_item.get("input", ""),
                "output": "Error generating recommendation",
                "error": str(e)
            }

    def _generate_recommendation(self, user_input: str, context: str = "") -> str:
        """Placeholder - Integrate your trained recommender model here"""
        # TODO: Load and use your trained model
        return f"Based on '{user_input}', I recommend: Jujutsu Kaisen, Demon Slayer, Attack on Titan"

    def run(self):
        """Run the evaluation"""
        try:
            logger.info(f"🚀 Starting Opik Evaluation: {self.experiment_name}")

            dataset = self.get_or_create_dataset()

            metrics = [Hallucination()]

            # Removed 'limit' parameter as it's not supported
            eval_results = evaluate(
                experiment_name=self.experiment_name,
                dataset=dataset,
                task=self.evaluation_task,
                scoring_metrics=metrics,
                verbose=True
            )

            logger.info("🎉 Evaluation completed successfully!")
            logger.info(f"Experiment Name: {self.experiment_name}")
            
            return eval_results

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise CustomException("Opik evaluation pipeline failed", e) from e


if __name__ == "__main__":
    try:
        evaluator = OpikEvaluator(
            experiment_name="anime-recommender-eval-v1",
            dataset_name="anime-recommendation-eval"
        )
        
        results = evaluator.run()
        
        print("\n✅ Evaluation finished successfully!")
        
    except Exception as e:
        logger.error(f"Failed to run evaluator: {e}")
        sys.exit(1)