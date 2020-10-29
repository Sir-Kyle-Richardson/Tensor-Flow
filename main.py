try:
    import argparse
    import interface
    import os
except ModuleNotFoundError:
    print('Program requires python module to be installed')
    exit(1)


def loadModeOption(argsMode):
    if argsMode in ["name", "name+", "series", "reset"]:
        return argsMode
    else:
        print("Wrong command option for mode")
        exit(1)


def runMode(modeOption, argsName, argsNumber, argsAnswer):
    if modeOption == "name":
        nameOfImage = loadArgParameter(argsName, "name")
        interface.runNameMode(nameOfImage)
    elif modeOption == "name+":
        nameOfImage = loadArgParameter(argsName, "name")
        classArgument = loadArgParameter(argsAnswer, "name+")
        interface.runNamePlusMode(nameOfImage, classArgument)
    elif modeOption == "series":
        numberOfImages = loadArgParameter(argsNumber, "series")
        interface.runSeriesMode(numberOfImages)
    elif modeOption == "reset":
        interface.runResetMode()


def loadArgParameter(argParameter, mode):
    if argParameter != None:
        return argParameter
    else:
        print(f"There is no enough parameters for {mode} mode")
        exit(1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Neural network for image recognition')
    parser.add_argument("-m", "--mode", type=str,
                        help="mode for program run: 'name' or 'name+' or 'series' or 'reset'")
    parser.add_argument("-i", "--number", type=int,
                        help="number of images named 'image_i.png' for series mode")
    parser.add_argument("-n", "--name", type=str,
                        help="name of image named e.g. 'image.png'")
    parser.add_argument("-a", "--answer", type=int,
                        help="correct class: '0,1,...,9' for the image in 'name+' mode")
    args = parser.parse_args()

    main_folder = os.getcwd()

    modeOption = loadModeOption(args.mode)
    runMode(modeOption, args.name, args.number, args.answer)
