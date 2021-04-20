from ..pipeline import Pipeline

from detectron2.engine import DefaultTrainer

from .utils.evaluator import VOCDetectionEvaluator


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return VOCDetectionEvaluator(dataset_name)


class TrainModel(Pipeline):
    """Pipeline task to register VOC annotations."""
    def __init__(self):
        """
        Init TrainModel instance
        """
        super(TrainModel,self).__init__()

    def map(self,data):
        """Train model specified in config."""
        trainer = Trainer(data)
        trainer.resume_or_load(resume=data.resume)
        return trainer.train()