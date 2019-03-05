import argparse
from utils.log_plotter import LogPlotter


def parse_arguments():
    parser = argparse.ArgumentParser(description="A script for plotting.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # General options
    dir_opts = parser.add_argument_group('directory options')
    dir_opts.add_argument('-l', '--log_dir', help='Path to the log file containing the data to plot.', required=True)
    dir_opts.add_argument('-o', '--output_file_name', help='The full path name to use when saving the plot.')

    args = parser.parse_args()
    return args


def main(args):
    LogPlotter(log_dir=args.log_dir, output_file_name=args.output_file_name)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
