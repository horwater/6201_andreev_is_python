import argparse
import json
import sys
import numpy as np
import matplotlib.pyplot as plt


# Загрузка конфиг-файла
def parse_config(filename):
    with open(filename, "r") as file:
        config = json.load(file)
    return config


def calculate_y(x, a, b, c):
    # Вычисление функции a * (2 * x + np.sin(b * x + c)**2) / (3 + x)
    return a * (2 * x + np.sin(b * x + c) ** 2) / (3 + x)


def main():
    parser = argparse.ArgumentParser(
        description="Calculate function y = a * (2*x + sin(b*x + c)^2) / (3 + x)"
    )
    parser.add_argument("--xmin", type=float, help="Minimum value of x")
    parser.add_argument("--step", type=float, help="Step size for x")
    parser.add_argument("--xmax", type=float, help="Maximum value of x")
    parser.add_argument("--a", type=float, help="Coefficient a")
    parser.add_argument("--b", type=float, help="Coefficient b")
    parser.add_argument("--c", type=float, help="Coefficient c")
    parser.add_argument(
        "--config", type=str, default="config.json", help="Path to config file (JSON)"
    )

    args = parser.parse_args()

    # Считывание параметров из конфиг-файла
    config = parse_config(args.config)

    # Переопределение параметров (на случай командной строки)
    if args.xmin is not None:
        config["xmin"] = args.xmin
    if args.step is not None:
        config["step"] = args.step
    if args.xmax is not None:
        config["xmax"] = args.xmax
    if args.a is not None:
        config["a"] = args.a
    if args.b is not None:
        config["b"] = args.b
    if args.c is not None:
        config["c"] = args.c

    # Проверка параметров
    required_params = ["xmin", "step", "xmax", "a", "b", "c"]
    for param in required_params:
        if param not in config:
            print(f"Error: Missing parameter '{param}' in config file or arguments")
            sys.exit(1)

    # Создание массива значений х
    x_values = np.arange(
        config["xmin"], config["xmax"] + config["step"], config["step"]
    )

    # Вычисление у для каждого значения х
    y_values = calculate_y(x_values, config["a"], config["b"], config["c"])

    # Запись результата
    with open("results", "w") as result_file:
        for x, y in zip(x_values, y_values):
            result_file.write(f"{x}\t{y}\n")

    # Изображение графика
    plt.figure(figsize=(8, 5))
    plt.plot(
        x_values,
        y_values,
        color="b",
        marker="o",
    )
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.title("График функции")
    plt.grid(True)
    plt.show()

# Вызов функции
main()
