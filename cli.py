from utils import get_training_args
import logging
from train import train
from eval import eval

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


if __name__ == "__main__":
    
    args = get_training_args()
    if args.train:
        logger.info(f"Training with the following arguments: {args}")
        train(args)
    
    if args.eval:
        logger.info(f"Evaluating with the following arguments: {args}")
        eval(args)
