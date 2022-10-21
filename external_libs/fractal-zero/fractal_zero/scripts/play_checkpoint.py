from argparse import ArgumentParser
import torch

from fractal_zero.fractal_zero import FractalZero


if __name__ == "__main__":
    # TODO: arg parser
    parser = ArgumentParser("play_checkpoint")
    parser.add_argument("checkpoint_path", type=str)

    args = parser.parse_args()

    trainer = torch.load(args.checkpoint_path)
    fractal_zero: FractalZero = trainer.fractal_zero

    fractal_zero.eval()
    fractal_zero.play_game(render=True)
