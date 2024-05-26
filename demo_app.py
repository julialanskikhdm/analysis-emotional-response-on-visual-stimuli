import csv
import os
import os.path
import sys
import tkinter
from tkinter import messagebox
from tkinter import *
import PIL
from PIL import Image, ImageTk
from tkinter import filedialog
from tkinter import ttk

import numpy as np
import torch
import torch.nn as nn
import customtkinter as tk
from PIL import Image, ImageDraw, ImageFont

import matplotlib.pyplot as plt
from datetime import datetime



from predict import *



tk.set_appearance_mode("dark")  # Modes: "System" (standard), "Dark", "Light"
tk.set_default_color_theme("dark-blue")  # Themes: "blue" (standard), "green", "dark-blue"

sys.path.append("..")


translated_img_name = ''

project_path_img = ''


def upload_photo():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        format_img = img.format
        global project_path_img
        project_path_img = f'uploaded_img/for_translation.{format_img}'

        img.save(project_path_img)

        img_resize = img.resize((300, 300))
        photo = ImageTk.PhotoImage(img_resize)
        label_ph = Label(image=photo)
        label_ph.image = photo
        label_ph.place(x=80, y=120)

# предсказание эмоциональной реакции
def analyzing_image():
    global max_emotion
    global max_probability
    distribution = predict_main(project_path_img)
    max_emotion = max(distribution, key=distribution.get)
    max_probability = distribution[max_emotion]
    tk.CTkLabel(win, text=f'{max_emotion}:', font=tk.CTkFont(size=14)).place(x=550, y=50)
    tk.CTkLabel(win, text=f'{max_probability}', font=tk.CTkFont(size=14)).place(x=665, y=50)

    labels = ['Amusement', 'Awe', 'Contentment', "Excitement", "Anger", "Disgust", "Fear", "Sadness"]
    values = list(distribution.values())

    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)

    for i, val in enumerate(values):
        plt.text(i, val, f'{val:.2f}', ha='center', va='bottom')

    plt.ylim(0, 1)
    plt.savefig('distribution_overall_analyze.png', dpi=60)
    plt.show()


def analyse_bright():
    pass

def analyze_color():
    pass

def analyze_face_expr():
    global class_face_expr
    class_face_expr, distribution = predict_face_expr(project_path_img)

    # tk.CTkLabel(win, text=f'{class_face_expr}', font=tk.CTkFont(size=16, weight="bold")).place(x=555, y=240)
    tk.CTkLabel(rectangle, text=f'{class_face_expr}', font=tk.CTkFont(size=14)).place(x=10, y=70)

def analyze_scene_type():
    probs, idx, classes = predict_scene(project_path_img)
    global class0, class1, class2, class3, class4, prob0, prob1, prob2, prob3, prob4
    class0 = classes[idx[0]]
    class1 = classes[idx[1]]
    class2 = classes[idx[2]]
    class3 = classes[idx[3]]
    class4 = classes[idx[4]]
    prob0 = probs[0]
    prob1 = probs[1]
    prob2 = probs[2]
    prob3 = probs[3]
    prob4 = probs[4]
    # tk.CTkLabel(win, text='{}: {:.3f}'.format(class0, prob0), font=tk.CTkFont(size=16)).place(x=555, y=280)
    # tk.CTkLabel(win, text='{}: {:.3f}'.format(class1, prob1), font=tk.CTkFont(size=16)).place(x=555, y=310)
    # tk.CTkLabel(win, text='{}: {:.3f}'.format(class2, prob2), font=tk.CTkFont(size=16)).place(x=555, y=340)
    # tk.CTkLabel(win, text='{}: {:.3f}'.format(class3, prob3), font=tk.CTkFont(size=16)).place(x=555, y=370)
    # tk.CTkLabel(win, text='{}: {:.3f}'.format(class4, prob4), font=tk.CTkFont(size=16)).place(x=555, y=400)
    tk.CTkLabel(rectangle, text='{}: {:.3f}'.format(class0, prob0), font=tk.CTkFont(size=14)).place(x=220, y=70)
    tk.CTkLabel(rectangle, text='{}: {:.3f}'.format(class1, prob1), font=tk.CTkFont(size=14)).place(x=220, y=100)
    tk.CTkLabel(rectangle, text='{}: {:.3f}'.format(class2, prob2), font=tk.CTkFont(size=14)).place(x=220, y=130)
    tk.CTkLabel(rectangle, text='{}: {:.3f}'.format(class3, prob3), font=tk.CTkFont(size=14)).place(x=220, y=160)
    tk.CTkLabel(rectangle, text='{}: {:.3f}'.format(class4, prob4), font=tk.CTkFont(size=14)).place(x=220, y=190)

def analyze_objects():
    global objects_list
    objects_list = predict_objects(project_path_img)
    y = 70
    for i in range(5):
        tk.CTkLabel(rectangle, text='{}'.format(objects_list[i]), font=tk.CTkFont(size=14)).place(x=430, y=y)
        y = y + 30


def check_report():
    # Создаем окно tkinter для отображения изображения
    root = tkinter.Toplevel()
    root.title("Generated Image")
    root.geometry("1000x2000")

    img_gen = ImageTk.PhotoImage(Image.open(img_gen_path))
    print("прошел открытие фотки")

    label_ph = tkinter.Label(root, image=img_gen)
    # label_ph.image = translated_sad_ph
    # label_ph.place(x=440, y=150)
    print("прошел создание label с фотографией")

    label_ph.place(x=5, y=5)
    print("прошел отобраение лейбла с фотой")

    root.mainloop()

def form_report():
    img = Image.new('RGB', (1000, 2500), color='white')
    d = ImageDraw.Draw(img)

    font_m = ImageFont.truetype("arial.ttf", 12)
    font = ImageFont.truetype("arial.ttf", 14)
    heading = "Report of Visual Emotion Analysis"

    bold_font_18 = ImageFont.truetype("arialbd.ttf", 18)
    bold_font_16 = ImageFont.truetype("arialbd.ttf", 16)
    bold_font_14 = ImageFont.truetype("arialbd.ttf", 14)
    d.text((400, 10), heading, fill='black', font=bold_font_18)

    now = datetime.now()
    copyright = ["PyCharm Community", f"{now}", "Lanskikh Julia"]
    d.text((750, 30), copyright[0], fill='black', font=font_m)
    d.text((750, 50), copyright[1], fill='black', font=font_m)
    d.text((750, 70), copyright[2], fill='black', font=font_m)

    # 1. Блок "Входное фото"
    text1 = "1. Uploaded image:"
    d.text((100, 90), text1, fill='black', font=bold_font_16)
    image_to_insert = Image.open(project_path_img)
    new_size = (350, 300)
    image_to_insert_resized = image_to_insert.resize(new_size)
    img.paste(image_to_insert_resized, (100, 120))

    # расстояние 50 между блоками

    # 2. Блок "Анализ эмоциональной реакции"
    text2 = "2. Emotional perception of the image and the confidence of the model:"
    d.text((100, 470), text2, fill='black', font=bold_font_16)
    results = f"{max_emotion}: {max_probability}"
    d.text((100, 500), results, fill='black', font=font)

    # 2.1 Блок "Распределение уверенности модели"
    text21 = "2.1. Distribution of model confidence:"
    d.text((100, 550), text21, fill='black', font=bold_font_14)
    image_to_insert = Image.open('distribution_overall_analyze.png')
    img.paste(image_to_insert, (100, 580))

    # 3 Блок "Классификация визуальных стимулов изображения"
    text3 = "3. Classification of visual image stimuli"
    d.text((100, 880), text3, fill='black', font=bold_font_16)
    # bright_class = f"3.1. Brightness: {match_br[class_br]}({class_br}): {prob_br}"
    # d.text((100, 910), bright_class, fill='black', font=font)
    # color_class = f"3.2. Colorfulness: {match_color[class_color]}({class_color}): {prob_color}"
    # d.text((100, 940), color_class, fill='black', font=font)
    face_expr_class = f"3.1. Face expression: {class_face_expr}"
    d.text((100, 910), face_expr_class, fill='black', font=font)

    scene_class = "3.2. Types of scenes:"
    d.text((100, 940), scene_class, fill='black', font=font)
    d.text((100, 970), '{}: {:.3f}'.format(class0, prob0), fill='black', font=font)
    d.text((100, 1000), '{}: {:.3f}'.format(class1, prob1), fill='black', font=font)
    d.text((100, 1030), '{}: {:.3f}'.format(class2, prob2), fill='black', font=font)
    d.text((100, 1060), '{}: {:.3f}'.format(class3, prob3), fill='black', font=font)
    d.text((100, 1090), '{}: {:.3f}'.format(class4, prob4), fill='black', font=font)

    object_class = "3.3. Object classes:"
    d.text((100, 1120), object_class, fill='black', font=font)
    objects_string = ", ".join(objects_list)
    d.text((100, 1150), objects_string, fill='black', font=font)
    image_to_insert = Image.open('object_classification.png')
    img.paste(image_to_insert, (100, 1170))

    # d.text((100, 1150), '{}: {:.3f}'.format(class4, prob4), fill='black', font=font)

    # возможно вставить сюда сводную таблицу по результатам классификации

    # 4 Блок "Предсказание эмоциональной реакции на основе визуальных стимулов изображения"
    text4 = "4. Emotional response analysis based on visual stimuli"
    d.text((100, 1570), text4, fill='black', font=bold_font_16)

    t41 = "4.1. Model based on facial expression:"
    d.text((100, 1600), t41, fill='black', font=bold_font_14)
    class_by_FE = f"{emotion_w_face_expr}: {max_prob_w_face_expr}"
    d.text((100, 1630), class_by_FE, fill='black', font=font)
    image_to_insert = Image.open('distrib_face_expr_stimuli_analyzing.png')
    img.paste(image_to_insert, (100, 1660))

    t42 = "4.2. Model based on scene types:"
    d.text((100, 2060), t42, fill='black', font=bold_font_14)
    class_by_Places = f"{emotion_w_scene}: {max_prob_w_scene}"
    d.text((100, 2090), class_by_FE, fill='black', font=font)
    image_to_insert = Image.open('distrib_scene_type_stimuli_analyzing.png')
    img.paste(image_to_insert, (100, 2120))
    # class_ST = f"3.4. {}: {}"
    # d.text((100, 1470), class_ST, fill='black', font=font)



    # Сохраняем изображение в формате jpg
    global img_gen_path
    img_gen_path = "example_report24.jpg"
    img.save(img_gen_path)


def analyse_w_face_expr_stimuli():

    global emotion_w_face_expr
    global max_prob_w_face_expr
    distribution = predict_emotion_w_face_expr(project_path_img)
    emotion_w_face_expr = max(distribution, key=distribution.get)
    max_prob_w_face_expr = distribution[emotion_w_face_expr]

    tk.CTkLabel(rectangle1, text=f'{emotion_w_face_expr}:', font=tk.CTkFont(size=14)).place(x=10, y=70)
    tk.CTkLabel(rectangle1, text=f'{max_prob_w_face_expr}', font=tk.CTkFont(size=14)).place(x=110, y=70)

    labels = ['Amusement', 'Awe', 'Contentment', "Excitement", "Anger", "Disgust", "Fear", "Sadness"]
    values = list(distribution.values())

    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)

    for i, val in enumerate(values):
        plt.text(i, val, f'{val:.2f}', ha='center', va='bottom')

    plt.ylim(0, 1)
    plt.savefig('distrib_face_expr_stimuli_analyzing.png', dpi=60)
    plt.show()

def analyse_w_scene_type_stimuli():
    global emotion_w_scene
    global max_prob_w_scene
    distribution = predict_emotion_w_scene_type(project_path_img)
    emotion_w_scene = max(distribution, key=distribution.get)
    max_prob_w_scene = distribution[emotion_w_scene]
    tk.CTkLabel(rectangle1, text=f'{emotion_w_scene}:', font=tk.CTkFont(size=14)).place(x=220, y=70)
    tk.CTkLabel(rectangle1, text=f'{max_prob_w_scene}', font=tk.CTkFont(size=14)).place(x=320, y=70)

    labels = ['Amusement', 'Awe', 'Contentment', "Excitement", "Anger", "Disgust", "Fear", "Sadness"]
    values = list(distribution.values())

    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)

    for i, val in enumerate(values):
        plt.text(i, val, f'{val:.2f}', ha='center', va='bottom')

    plt.ylim(0, 1)
    plt.savefig('distrib_scene_type_stimuli_analyzing.png', dpi=60)
    plt.show()

def analyse_w_bright_stimuli():
    pass

def analyse_w_color_stimuli():
    pass

def on_closing():
    if messagebox.askokcancel("Exit", "Do you want to exit?"):
        win.destroy()

def create_bar_chart():
    pass

global win
# win = tk.Tk()
win = tk.CTk()


win.protocol("WM_DELETE_WINDOW", on_closing)

win.geometry("750х900")
win.resizable(True, True)
win.title('Visual stimuli analyzer')

# CHECK: расположить надпись центральную ("Emotional Response") по середине
tk.CTkLabel(win, text='Analyzing Emotional Response and Visual Stimuli', font=tk.CTkFont(size=20, weight="bold")).place(x=95, y=10)
# tk.CTkLabel(win, text='Upload image', font=(12, 'bold')).place(x=120, y=35)
upload_button = tk.CTkButton(win, text='Upload image', command=upload_photo)
upload_button.place(x=120, y=50)


# кнопка с предсказанием эмоциональной реакцией + генерация графика
analyse_button = tk.CTkButton(win, text='Analyse Image',  command=analyzing_image)
analyse_button.place(x=405, y=50)



# распознавание коэфф яркости
# brightness_button = tk.CTkButton(win, text='Brightness',   command=analyse_bright)
# brightness_button.place(x=405, y=160)

# распознавание коэф насыщенности
# colorfulness_button = tk.CTkButton(win, text='Colorfulness',   command=analyze_color)
# colorfulness_button.place(x=405, y=200)


# Создаем прямоугольник
global rectangle
rectangle = tk.CTkCanvas(win, width=720, height=300, bg="black", highlightthickness=1, highlightbackground="blue")
rectangle.place(x=505, y=150)

tk.CTkLabel(rectangle, text='Classification visual stimuli', font=tk.CTkFont(size=16, weight="bold")).place(x=10, y=7)

# распознавание эмоции лиц на изображении
face_expr_button = tk.CTkButton(rectangle, text='Face expression',  command=analyze_face_expr)
# face_expr_button.place(x=405, y=240)
face_expr_button.place(x=10, y=40)

# распознавание эмоции лиц на изображении
scene_button = tk.CTkButton(rectangle, text='Scene type',  command=analyze_scene_type)
# scene_button.place(x=405, y=280)
# scene_button.place(x=405, y=200)
scene_button.place(x=220, y=40)

# распознавание объектов на изображении (OpenImagesV4)
objects_classification_button = tk.CTkButton(rectangle, text='Objects',  command=analyze_objects)
# objects_classification_button.place(x=405, y=280)
objects_classification_button.place(x=430, y=40)


global rectangle1
rectangle1 = tk.CTkCanvas(win, width=720, height=150, bg="black", highlightthickness=1, highlightbackground="blue")
rectangle1.place(x=505, y=500)

tk.CTkLabel(rectangle1, text='Analysis with visual stimuli', font=tk.CTkFont(size=16, weight="bold")).place(x=10, y=7)

# предсказание реакции с учетом коэфф яркости
# brightness_button = tk.CTkButton(win, text='Brightness',   command=analyse_w_bright_stimuli)
# brightness_button.place(x=405, y=480)

# предсказание реакции с учетом  коэф насыщенности
# colorfulness_button = tk.CTkButton(win, text='Colorfulness',   command=analyse_w_color_stimuli)
# colorfulness_button.place(x=405, y=520)

# предсказание реакции с учетом эмоции лиц на изображении
face_expr_button = tk.CTkButton(rectangle1, text='Face expression',  command=analyse_w_face_expr_stimuli)
# face_expr_button.place(x=405, y=560)
face_expr_button.place(x=10, y=40)

# предсказание реакции с учетом типов сцены на изображении
scene_type_button = tk.CTkButton(rectangle1, text='Scene type',  command=analyse_w_scene_type_stimuli)
# scene_type_button.place(x=405, y=600)
scene_type_button.place(x=220, y=40)

global rectangle2
rectangle2 = tk.CTkCanvas(win, width=720, height=105, bg="black", highlightthickness=1, highlightbackground="blue")
rectangle2.place(x=505, y=700)

tk.CTkLabel(rectangle2, text='Report with results', font=tk.CTkFont(size=16, weight="bold")).place(x=10, y=7)

# выгрузка отчета
generate_report_button = tk.CTkButton(rectangle2, text='Generate report',   command=form_report)
# report_button.place(x=405, y=630)
generate_report_button.place(x=10, y=40)

# просмотр отчета
report_check_button = tk.CTkButton(rectangle2, text='Download report',   command=check_report)
# report_check_button.place(x=405, y=660)
report_check_button.place(x=220, y=40)



win.mainloop()

