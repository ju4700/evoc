def sum_squares(n):
    out = []
    for i in range(n):
        out.append(i * i)
    return sum(out)

if __name__ == '__main__':
    print(sum_squares(10))
