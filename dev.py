import sys
sys.path.insert(0, "avalanche")

from benchmarks import get_cifar_based_benchmark

benchmark = get_cifar_based_benchmark("config_s3.pkl",0)

print(benchmark.train_stream[0].dataset)