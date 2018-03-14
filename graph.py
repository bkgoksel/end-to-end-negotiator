import matplotlib.pyplot as plt

def main():
    cols = None
    rows = []
    with open('eps.tsv', 'r') as f:
        for line in f:
            if cols is None:
                cols = line.split(' ')
            else:
                vals = [float(val) for val in line.split(' ')]
                rows.append(vals)

    x = [row[0] for row in rows]
    for idx, col in enumerate(cols):
        if idx == 0:
            continue
        y = [row[idx] for row in rows]
        plt.plot(x, y)
        plt.title(col.replace('_', ' '))
        plt.show()
        plt.savefig(col.strip() +'.png')
        plt.clf()

if __name__ == '__main__':
    main()
