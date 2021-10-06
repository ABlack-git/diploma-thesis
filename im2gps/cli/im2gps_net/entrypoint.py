import click

from im2gps.cli.im2gps_net.network import train, tune, test


@click.group()
def entrypoint():
    pass


entrypoint.add_command(train)
entrypoint.add_command(tune)
entrypoint.add_command(test)

if __name__ == '__main__':
    entrypoint()
