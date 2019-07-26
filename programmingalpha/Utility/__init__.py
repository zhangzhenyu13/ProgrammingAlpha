import logging

def getLogger(__name__):
        logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt = '%m/%d/%Y %H:%M:%S',
                            level = logging.INFO)

        logger = logging.getLogger(__name__)

        return logger


