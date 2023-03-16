import random
def spatial(feature, category_a=1, category_b=2):
    [spatial, _] = feature
    epsilon = 0
    criterion = 50
    discriminant = spatial - criterion
    return (category_a if discriminant > epsilon else category_b), abs(discriminant)


def spatial_and_orientation(feature, category_a=1, category_b=2):
    [spatial, ori] = feature
    epsilon = 0
    criterion = 50
    spa_discriminant = spatial - criterion
    ori_discriminant = ori - criterion
    return (category_a, abs(spa_discriminant))if (spa_discriminant > epsilon) and (ori_discriminant > epsilon) else (category_b, abs(ori_discriminant))


def orientation(feature, category_a=1, category_b=2):
    [_, ori] = feature
    epsilon = 0
    criterion = 50
    ori_discriminant = ori - criterion
    return (category_a if ori_discriminant > epsilon else category_b), abs(ori_discriminant)
