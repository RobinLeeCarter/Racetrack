import controller
import os


def main():
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
    controller_ = controller.Controller(verbose=False)
    controller_.run()


if __name__ == '__main__':
    main()
