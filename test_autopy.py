import autopy

WIDTH, HEIGHT = (int(x) for x in autopy.screen.size())


def move():
    for i in range(0, WIDTH, 10):
        for j in range(0, HEIGHT, 10):
            # print(f"MOVE TO ({i}, {j})")
            autopy.mouse.move(i, j)


if __name__ == '__main__':
    move()
