import sys
from PySide6.QtWidgets import QApplication
from annotator import YOLOXCOCOAnnotator


def main():
    app = QApplication(sys.argv)
    window = YOLOXCOCOAnnotator()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
