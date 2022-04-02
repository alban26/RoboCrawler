import gui.crawler_gui
import logging


def main():
    logging.basicConfig(level=logging.DEBUG)
    gui.crawler_gui.start()


if __name__ == '__main__':
    main()
