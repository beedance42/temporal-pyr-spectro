"""
This program allows for the creation of 'temporal pyramids' from
webcam video datasets or datasets of static-vantage-point
time series images.
"""
import argparse

import inputdata
import buildpyr

def get_args():
    """
    Parse the command line arguments.
    Output: tuple with all parameter values:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "datasource",
        help = "name of datasource object (GOES, Geiranger, Chattahoochee)")
    parser.add_argument(
        "method",
        help = "method for processing input (pyr-1-day, pyr-multi-day,\
                pyr-multi-year, reconstruct, stitch)")
    parser.add_argument(
        "-date",
        help = "date in yymmdd format, used for most methods (default=\"200724\")",
        default = "200724")
    parser.add_argument(
        "-top",
        help = "top level to start reconstruction from (default=15)",
        default = 15,
        type = int)
    parser.add_argument(
        "-bot",
        help = "bottom level to stop reconstruction at (default=0)",
        default = 0,
        type = int)
    parser.add_argument(
        "-sub",
        help = "Levels to subtract when doing reconstruction (default is empty list)",
        default = [],
        nargs = '+',
        type = int)
    parser.add_argument(
        "-lv",
        help = "level number to use for method (default is 14)",
        default = 14,
        type = int)
    parser.add_argument(
        "-yr",
        help = "year for use with pyr-multi-day function (YYYY)",
        default = None,
        type = int)

    args = parser.parse_args()
    return (
            args.datasource,
            args.method,
            args.date,
            args.top,
            args.bot,
            args.sub,
            args.lv,
            args.yr
    )


def main():
    """
    Parse command line arguments and run selected algorithm.
    """
    datasource, method, date, top, bottom, subtract, level, year = get_args()

    if method == "pyr-1-day":
        input_data = inputdata.InputDataOneDay(datasource, date)
        buildpyr.laplacian_temporal_pyr(input_data)
    elif method == "pyr-multi-day":
        input_data = inputdata.InputData(datasource)
        buildpyr.create_pyr_multiday(input_data, year)
    elif method == "pyr-multi-year":
        input_data = inputdata.InputData(datasource)
        buildpyr.create_pyr_multiyear(input_data)
    elif method == "reconstruct":
        input_data = inputdata.InputData(datasource)
        buildpyr.reconstruct_pyr(input_data, top, bottom, subtract)
    elif method == "stitch":
        input_data = inputdata.InputData(datasource)
        buildpyr.stitch_level(input_data, level)

if __name__ == "__main__":
    main()
