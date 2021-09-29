import click

from im2gps.cli.im2gps_net.network import train


@click.group()
def entrypoint():
    pass


entrypoint.add_command(train)

if __name__ == '__main__':
    entrypoint()
