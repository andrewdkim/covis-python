import random
def spatial(entry, category_a=1, category_b=2):
    [_, spatial, _] = entry
    epsilon = random.randint(-15, 15)
    criterion = 22.5
    return category_a if (spatial - criterion) > epsilon else category_b


def spatial_and_orientation(entry, category_a=1, category_b=2):
    [_, spatial, ori] = entry
    epsilon = random.randint(-15, 15)
    criterion = 22.5
    return category_a if ((spatial - criterion) > epsilon) and ((ori - criterion) > epsilon) else category_b


def orientation(entry, category_a=1, category_b=2):
    [_, _, ori] = entry
    epsilon = random.randint(-15, 15)
    criterion = 22.5
    return category_a if (ori - criterion) > epsilon else category_b
