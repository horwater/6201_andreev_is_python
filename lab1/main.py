import argparse
import json
import sys

def parse_config(filename):
    with open(filename, 'r') as file:
        config = json.load(file)
    return config

def calculate_y(x, a, b, c):
    return a * x**2 + b * x + c

def main():
    parser = argparse.ArgumentParser(description='Calculate function y = a*x^2 + b*x + c')
    parser.add_argument('--xmin', type=float, help='Minimum value of x')
    parser.add_argument('--step', type=float, help='Step size for x')
    parser.add_argument('--xmax', type=float, help='Maximum value of x')
    parser.add_argument('--a', type=float, help='Coefficient a')
    parser.add_argument('--b', type=float, help='Coefficient b')
    parser.add_argument('--c', type=float, help='Coefficient c')
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file (JSON)')

    args = parser.parse_args()

    # Считываем параметры из конфигурационного файла
    config = parse_config(args.config)

    # Переопределяем параметры, если они переданы через аргументы командной строки
    if args.xmin is not None:
        config['xmin'] = args.xmin
    if args.step is not None:
        config['step'] = args.step
    if args.xmax is not None:
        config['xmax'] = args.xmax
    if args.a is not None:
        config['a'] = args.a
    if args.b is not None:
        config['b'] = args.b
    if args.c is not None:
        config['c'] = args.c

    # Проверяем, что все параметры заданы
    required_params = ['xmin', 'step', 'xmax', 'a', 'b', 'c']
    for param in required_params:
        if param not in config:
            print(f"Error: Missing parameter '{param}' in config file or arguments")
            sys.exit(1)

    # Вычисляем значения функции и записываем их в файл
    with open('results', 'w') as result_file:
        x = config['xmin']
        while x <= config['xmax']:
            y = calculate_y(x, config['a'], config['b'], config['c'])
            result_file.write(f"{x}\t{y}\n")
            x += config['step']

if __name__ == "__main__":
    main()
