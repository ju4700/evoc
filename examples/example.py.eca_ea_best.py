def sum_squares(n):
    out = [i * i for i in range(n)]
    return sum(out)
if __name__ == '__main__':
    print(sum_squares(10))