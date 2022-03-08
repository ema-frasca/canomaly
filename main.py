import datetime
from uuid import uuid4

from utils.args import parse_args
from utils.config import config
from datasets import get_dataset
from models import get_model
import wandb


# conf_path = os.getcwd()
# sys.path.append(conf_path)


def main(args=None):
    if args is None:
        args = parse_args()

    if args.seed is not None:
        config.set_seed(args.seed)
    else:
        args.seed = config.seed

    args.id = str(uuid4())
    args.timestamp = str(datetime.datetime.now())

    dataset = get_dataset(args.dataset)(args)
    model = get_model(args.model)(args, dataset)

    if args.wandb:
        wandb.init(project="canomaly", entity="ema-frasca", config={**vars(args)})

    model.train_on_dataset()
    model.print_results()


if __name__ == '__main__':
    main()
