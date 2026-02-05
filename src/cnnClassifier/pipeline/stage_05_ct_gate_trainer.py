from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.ct_gate_trainer import CTGateTrainer
from cnnClassifier import logger


STAGE_NAME = "CT Gate Trainer stage"

class CTGateTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        ct_gate_config = config.get_ct_gate_config()
        trainer = CTGateTrainer(ct_gate_config)
        trainer.train()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = CTGateTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
        print("https://dagshub.com/sujalvairagi/sujalvairagi-CI-CD_chest_cancer_with_vgg-16.mlflow/#/experiments/0/runs?searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=LAST_24_HOURS&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D")
    except Exception as e:
        logger.exception(e)
        raise e

