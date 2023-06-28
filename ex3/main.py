from train import train_model
from train import test
from train import plot_result

def main():
    best_weight = train_model()
    test(best_weight)
    plot_result()


if __name__ == '__main__':
    main()
