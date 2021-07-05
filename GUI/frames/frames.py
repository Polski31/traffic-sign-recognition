import tkinter as tk
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory
from recognition.predict import Predict
from cv2 import imread
from cv2 import imshow

LARGE_FONT = ("Verdana", 12)


class MainPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Menu glowne", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        button1 = tk.Button(self, text="Sprawdz znak",
                            command=lambda: controller.show_frame(PredictPage))
        button1.pack()

        button2 = tk.Button(self, text="Pokaz znak",
                            command=lambda: controller.show_frame(ShowImagePage))
        button2.pack()


class PredictPage(tk.Frame):

    def __init__(self, parent, controller):
        self.model_path = ""
        self.image_dir_path = ""
        self.csv = ""
        self.examples_path = ""

        tk.Frame.__init__(self, parent)

        label = tk.Label(self, text="Sprawdz znak", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        button1 = tk.Button(self, text="Menu glowne",
                            command=lambda: controller.show_frame(MainPage))
        button1.pack()

        button2 = tk.Button(self, text="Wybierz model",
                            command=lambda: self.open_model())
        button2.pack()

        button3 = tk.Button(self, text="Wybierz folder wyniku",
                            command=lambda: self.open_examples())
        button3.pack()

        button4 = tk.Button(self, text="Wybierz plik csv",
                            command=lambda: self.open_csv())
        button4.pack()

        button5 = tk.Button(self, text="Wybierz folder do testu",
                            command=lambda: self.open_test_image())
        button5.pack()

        button6 = tk.Button(self, text="Potwierz",
                            command=lambda: self.recognize_sign())
        button6.pack()

    def open_model(self):
        model_path = askdirectory(title="Wybierz model.")
        print(model_path)
        self.model_path = model_path

    def open_examples(self):
        examples_path = askdirectory(title="Wybierz folder ze zdjeciami.")
        print(examples_path)
        self.examples_path = examples_path

    def open_csv(self):
        csv_path = askopenfilename(title="Wybierz plik csv",
                                   filetypes=[("CSV file", "*.csv")])
        print(csv_path)
        self.csv = csv_path

    def open_test_image(self):
        image_dir_path = askdirectory(title="Wybierz folder ze zdjeciami.")
        print(image_dir_path)
        self.image_dir_path = image_dir_path

    def recognize_sign(self):
        # print(self.model_path + self.csv + self.image_dir_path, sep="    ")
        Predict(self.model_path, self.examples_path, self.csv, self.image_dir_path).predict()


class ShowImagePage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Menu glowne", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        button1 = tk.Button(self, text="Menu glowne",
                            command=lambda: controller.show_frame(MainPage))
        button1.pack()

        button2 = tk.Button(self, text="Wybierz zdjecie",
                            command=lambda: self.select_image())
        button2.pack()

    @staticmethod
    def select_image():
        image_path = askopenfilename(title="Select image",
                                     filetypes=[("Images", "*.png")])
        x = imread(image_path)
        imshow("Znak", x)
