from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.days_to_death_regressor import DaysToDeathRegressor
from cnnClassifier import logger


STAGE_NAME = "Days-to-Death Regression stage"


class DaysToDeathRegressionPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        dtd_config = config.get_days_to_death_config()
        regressor = DaysToDeathRegressor(dtd_config)
        regressor.train()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DaysToDeathRegressionPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
