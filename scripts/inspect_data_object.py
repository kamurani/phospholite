import torch 
import click as ck
import glob

from pathlib import Path


@ck.command()
@ck.option(
    "--path",
    "-p",
    help="Path to data object.",
)
def main(
    path: str,
):
    """Inspect a data object."""
    path = Path(path)
    data = torch.load(path)
    print(data)
          
    
if __name__ == "__main__":
    main()
