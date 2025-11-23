import tkinter as tk
from tkinter import *

# ----- Main Window -----
root = Tk()
root.title("Professional Calculator")
root.geometry("450x600")
root.resizable(False, False)
root.configure(bg="#2C2C2C")

# ----- Entry Display -----
display = Entry(root, font=("Arial", 28), bd=0, bg="#3C3C3C", fg="white",
                justify=RIGHT, insertbackground="white")
display.pack(fill=X, ipadx=8, ipady=20, padx=10, pady=10)


# ----- Button Function -----
def on_click(value):
    if value == "C":
        display.delete(0, END)
    elif value == "=":
        try:
            result = eval(display.get())
            display.delete(0, END)
            display.insert(END, result)
        except:
            display.delete(0, END)
            display.insert(END, "Error")
    else:
        display.insert(END, value)


# ----- Button Style -----
def create_button(parent, text, bg, fg, r, c, w=5):
    btn = Button(parent, text=text, font=("Arial", 20), bg=bg, fg=fg,
                 activebackground="#555555", activeforeground="white",
                 width=w, height=2, bd=0,
                 command=lambda: on_click(text))
    btn.grid(row=r, column=c, padx=5, pady=5)
    return btn


# ----- Frame for Buttons -----
button_frame = Frame(root, bg="#2C2C2C")
button_frame.pack()

# ----- Buttons Layout -----
buttons = [
    ["C", "/", "*", "-"],
    ["7", "8", "9", "+"],
    ["4", "5", "6", "="],
    ["1", "2", "3", ""],
    ["0", ".", "", ""]
]

row = 0
for r in buttons:
    col = 0
    for text in r:
        if text != "":
            if text in ["+", "-", "*", "/", "="]:
                create_button(button_frame, text, "#FF9500", "white", row, col)
            elif text == "C":
                create_button(button_frame, text, "#D32F2F", "white", row, col)
            else:
                create_button(button_frame, text, "#4E4E4E", "white", row, col)
        col += 1
    row += 1

root.mainloop()
