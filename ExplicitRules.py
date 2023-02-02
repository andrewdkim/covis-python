import random
def spatial(entry, category_a=1, category_b=2):
    [_, spatial, _] = entry
    epsilon = 0
    # criterion = 22.5
    criterion = 45
    discriminant = spatial - criterion
    # spa_maximum = 101.25
    # spa_minimum = -11.25
    spa_maximum = 50 # 95 - 45 = 50
    spa_minimum = -40 # 5 - 45 = -40
    #norm = disc + abs(min) / max + abs(min)
    return (category_a if discriminant > epsilon else category_b),  abs((discriminant + abs(spa_minimum)) / (abs(spa_minimum) + spa_maximum))


def spatial_and_orientation(entry, category_a=1, category_b=2):
    [_, spatial, ori] = entry
    epsilon = 0
    # criterion = 22.5
    criterion = 45
    spa_discriminant = spatial - criterion
    ori_discriminant = ori - criterion
    maximum = 50 # 95 - 45 = 50
    minimum = -40 # 5 - 45 = -40
    ori_conf = abs((ori_discriminant + abs(maximum)) / (abs(minimum) + maximum))
    spa_conf = abs((spa_discriminant + abs(maximum)) / (abs(minimum) + maximum))
    return (category_a if (spa_discriminant > epsilon) and (ori_discriminant > epsilon) else category_b), abs((ori_conf + spa_conf) / 2)


def orientation(entry, category_a=1, category_b=2):
    [_, _, ori] = entry
    epsilon = 0
    criterion = 45
    ori_discriminant = ori - criterion
    maximum = 50 # 95 - 45 = 50
    minimum = -40 # 5 - 45 = -40
    return (category_a if ori_discriminant > epsilon else category_b), abs((ori_discriminant + abs(minimum)) / (abs(minimum) + maximum))
